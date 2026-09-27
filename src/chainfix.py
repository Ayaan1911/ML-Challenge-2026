"""Pass 8c: chain-name candidate fix (docs/france_diagnosis.md). Chain names ("bordeaux club" x307 S1) are skipped by
the capped exact keys and tie in the top-5 name retriever, so some S1 end up with no candidate at all although a
record with the same name and house number exists.

Rule (same on the eval side and on test): for every scored S1 with no candidate, add as candidates the records of
its country that already have candidates and share its exact name_core and first house number (uncapped). The
new pairs get the blocking columns exactly as src.blocking computes them: key flags from the same KeyIndex caps,
true TF-IDF cosines from the same per-country S1 indexes, and addr_word_idf = 0 (the rare-word retriever did not
return them). Each such record's rows (old plus new) are then re-featurized and re-scored. Record context depends
only on the record's own rows, and the target S1 had no rows before, so no other row's features change.

Two self-checks abort a run: recomputed blocking columns must equal the stored ones on 2,000 existing pairs, and
300 untouched control records re-featurized the same way must reproduce their saved probabilities exactly.

Run: python -m src.chainfix eval              -> eval corrected F0.5 before/after (docs/pass8c_variants.md numbers)
     python -m src.chainfix test france|india -> data/cache/pred_test_chainfix/<country>.parquet (delta only;
                                                 data/cache/pred_test/ is never written)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel

from src.blocking import ALL_KEYS, ALWAYS_KEYS, CONFIG, MULTI_KEYS, SCORES, KeyIndex, TfidfIndex, load_norm, prune_score
from src.data import CACHE, ROOT, id_key, key_id, load_truth, peak_rss_mb, start_memory_guard
from src.matcher import NormTable, _grouped_rank, featurize, load_model, part_country, part_paths, predict, read_part
from src.norm_cache import norm_path
from src.split import SPLITS

DELTA_DIR = CACHE / "pred_test_chainfix"
SPLIT = {"val": "train", "test": "test"}
rng = np.random.default_rng(42)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


def name_num(split: str, source_ids: list[tuple[str, list[str] | None]], country: str) -> pd.DataFrame:
    """entity_id, name_core, first house number for the given sources (ids=None: the whole country)."""
    frames = []
    for s, ids in source_ids:
        flt = [("country_norm", "=", country)] + ([("entity_id", "in", ids)] if ids is not None else [])
        t = pq.read_table(norm_path(split, s), columns=["entity_id", "name_core", "numbers"], filters=flt).to_pandas()
        t["num"] = [n[0] if len(n) else "" for n in t.pop("numbers")]
        frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    return d[(d["name_core"] != "") & (d["num"] != "")]


def new_pairs(split: str, country: str, targets: list[str], pool: list[str]) -> pd.DataFrame:
    """(s1_id, cand_id) for target S1 x pool records with identical name_core and first house number."""
    s1 = name_num(split, [("source1", targets)], country)
    rec = name_num(split, [("source2", None), ("source3", None)], country)
    rec = rec[rec["entity_id"].isin(set(pool))]
    pairs = s1.merge(rec, on=["name_core", "num"], suffixes=("_s1", "_rec"))
    return pd.DataFrame({"s1_id": pairs["entity_id_s1"].to_numpy(object), "cand_id": pairs["entity_id_rec"].to_numpy(object)})


def block_cols(split: str, country: str, pairs: pd.DataFrame) -> pd.DataFrame:
    """Blocking columns for arbitrary (s1_id, cand_id) pairs of one country, computed as CountryBlocker does."""
    cfg = CONFIG
    s1 = load_norm(split, ("source1",), country)
    rec = load_norm(split, ("source2", "source3"), country, ids=sorted(set(pairs["cand_id"])))
    s_idx = pd.Index(s1["entity_id"]).get_indexer(pairs["s1_id"])
    r_idx = pd.Index(rec["entity_id"]).get_indexer(pairs["cand_id"])
    if (s_idx < 0).any() or (r_idx < 0).any():
        raise SystemExit(f"{country}: pair ids missing from the norm cache")
    n1 = len(s1)
    want = r_idx.astype(np.int64) * n1 + s_idx
    out = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(object), "cand_id": pairs["cand_id"].to_numpy(object)})
    own = np.arange(n1)
    for f in ALL_KEYS:
        cap = cfg["block_cap"] if f in ALWAYS_KEYS else cfg["new_key_cap"]
        index = KeyIndex(*MULTI_KEYS[f](s1), cap) if f in MULTI_KEYS else KeyIndex(s1[f].to_numpy(object), own, cap)
        keys, labels = MULTI_KEYS[f](rec) if f in MULTI_KEYS else (rec[f].to_numpy(object), np.arange(len(rec)))
        r, s = index.lookup(keys, labels)
        out[f] = np.isin(want, r.astype(np.int64) * n1 + s).astype(np.int8)
        del index
    cfg = {**cfg, "threads": 1}  # hash in-process: 6 loky workers held ~2 GB of text copies next to this process
    with Parallel(n_jobs=1) as par:
        for col, texts_s1, texts_rec, prefix in (
                ("tfidf_name_cos", s1["name_core"], rec["name_core"], cfg["prefix_name"]),
                ("tfidf_nameaddr_cos", (s1["name_core"] + " " + s1["addr_core"]).str.strip(),
                 (rec["name_core"] + " " + rec["addr_core"]).str.strip(), cfg["prefix_nameaddr"])):
            index = TfidfIndex(texts_s1.tolist(), cfg, par, prefix, probe=False)
            out[col] = index.cosine(index.vectorize(texts_rec.tolist()), r_idx, s_idx)
            del index
    out["addr_word_idf"] = np.float32(0)
    return out


def check_block_cols(kind: str, country: str, paths: list) -> None:
    """Recompute the blocking columns of 2,000 stored pairs; key flags must match exactly, cosines to 1e-5."""
    df = pd.read_parquet(paths[len(paths) // 2])
    df = df.iloc[rng.choice(len(df), min(2000, len(df)), replace=False)].reset_index(drop=True)
    got = block_cols(SPLIT[kind], country, df[["s1_id", "cand_id"]])
    bad = [f for f in ALL_KEYS if (got[f].to_numpy() != df[f].to_numpy()).any()]
    err = max(float(np.abs(got[c].to_numpy() - df[c].to_numpy()).max()) for c in SCORES[:2])
    if bad or err > 1e-5:
        raise SystemExit(f"{country}: recomputed blocking columns differ from stored ones (keys {bad}, cos err {err})")
    log(f"{country}: blocking columns of 2,000 stored pairs reproduced (max cosine error {err:.1e})")


def rescore(kind: str, country: str, new: pd.DataFrame, controls: int = 300) -> pd.DataFrame:
    """All rows of the records in `new` (stored + new) re-featurized and scored, plus control records.
    new: blocking columns + meta of the new pairs. Returns rows with prob and an is_control flag."""
    paths = [p for p in part_paths(kind) if part_country(p) == country]
    affected = np.unique(id_key(new["cand_id"].to_numpy(object)))
    s1s, scores, weights, rows, pos, off = [], [], [], [], [], 0
    ctrl_pool = []
    for p in paths:  # one light pass: S1 context inputs of the whole country + the affected records' rows
        df = read_part(kind, p)
        rk = id_key(df["cand_id"].to_numpy(object))
        s1s.append(id_key(df["s1_id"].to_numpy(object)))
        scores.append(df["prune_score"].to_numpy(np.float32))
        weights.append(df["weight"].to_numpy(np.float32))
        pool = rk[~np.isin(rk, affected) & ((df["side"] == "eval").to_numpy() if "side" in df else True)]
        ctrl_pool.append(rng.choice(np.unique(pool), controls // len(paths) + 1, replace=False))  # saved preds: eval side
        m = np.isin(rk, np.r_[affected, ctrl_pool[-1]])
        rows.append(df[m])
        pos.append(off + np.flatnonzero(m))
        off += len(df)
    new = new.assign(prune_score=prune_score(new))
    s1 = np.concatenate(s1s + [id_key(new["s1_id"].to_numpy(object))])
    inv = np.unique(s1, return_inverse=True)[1]
    del s1, s1s
    score = np.concatenate(scores + [new["prune_score"].to_numpy(np.float32)])
    weight = np.concatenate(weights + [new["weight"].to_numpy(np.float32)])
    n_cands = np.bincount(inv, weights=weight).astype(np.float32)[inv]
    rank = _grouped_rank(inv, score, weight)
    idx = np.concatenate(pos + [np.arange(off, off + len(new))])
    df = pd.concat(rows + [new[rows[0].columns]], ignore_index=True)
    ctx = pd.DataFrame({"s1_n_cands": n_cands[idx], "s1_rank": rank[idx]})
    del inv, score, weight, n_cands, rank
    s1k, rk = id_key(df["s1_id"].to_numpy(object)), id_key(df["cand_id"].to_numpy(object))
    f = featurize(df, NormTable(SPLIT[kind], ("source1",), s1k), NormTable(SPLIT[kind], ("source2", "source3"), rk), ctx)
    df["prob"] = predict(load_model(), f)
    df["is_control"] = ~np.isin(rk, affected)
    df["is_new"] = np.r_[np.zeros(len(df) - len(new), bool), np.ones(len(new), bool)]
    log(f"{country}: re-scored {len(df):,} rows ({len(new):,} new pairs, {len(affected):,} records, "
        f"{int(df['is_control'].sum()):,} control rows)")
    return df


def check_controls(df: pd.DataFrame, saved: pd.Series) -> None:
    c = df[df["is_control"]]
    got = saved.reindex(pd.MultiIndex.from_arrays([id_key(c["s1_id"].to_numpy(object)),
                                                   id_key(c["cand_id"].to_numpy(object))])).to_numpy()
    err = float(np.nanmax(np.abs(c["prob"].to_numpy() - got))) if len(c) else 0.0
    if np.isnan(got).any() or err > 0:
        raise SystemExit(f"control records do not reproduce their saved probabilities (max error {err})")
    log(f"{len(c):,} control rows reproduce the saved probabilities exactly")


# ---------- eval side ----------

def run_eval() -> None:
    """The prediction table is held only before and after the per-country work (memory)."""
    from src.postprocess import Eval, PRED_EVAL
    t0 = time.perf_counter()
    b_ids = (SPLITS / "valsplit_B_s1_ids.txt").read_text(encoding="utf-8").split()
    truth = load_truth()
    s1c = pq.read_table(norm_path("train", "source1"), columns=["entity_id", "country_norm"],
                        filters=[("entity_id", "in", b_ids)]).to_pandas().set_index("entity_id")["country_norm"]
    pe = pd.read_parquet(PRED_EVAL, columns=["s1_id", "cand_id", "country", "query_group"])
    todo = {}
    for country in ("india", "us"):
        c_pe = pe[pe["country"] == country]
        todo[country] = (sorted(set(s1c.index[s1c == country]) - set(c_pe["s1_id"])), c_pe["cand_id"].unique().tolist(),
                         c_pe.drop_duplicates("cand_id").set_index("cand_id")["query_group"])
    del pe, c_pe
    deltas, controls, stats = [], [], {}
    for country, (targets, pool, groups) in todo.items():
        paths = [p for p in part_paths("val") if part_country(p) == country]
        check_block_cols("val", country, paths)
        blk = block_cols("train", country, new_pairs("train", country, targets, pool))
        blk["country"] = country
        blk["query_group"] = groups.reindex(blk["cand_id"]).to_numpy(object)
        blk["weight"] = np.where(blk["query_group"] == "other_sampled", 5.0, 1.0).astype(np.float32)
        blk["side"] = "eval"
        blk["label"] = np.array([r in truth[s] for s, r in zip(blk["s1_id"], blk["cand_id"])], np.int8)
        df = rescore("val", country, blk)
        stats[country] = {"target_s1": len(targets), "target_s1_with_new_pair": int(blk["s1_id"].nunique()),
                          "new_pairs": len(blk), "new_true_pairs": int(blk["label"].sum()),
                          "records_rescored": int(df.loc[~df["is_control"], "cand_id"].nunique())}
        cols = ["s1_id", "cand_id", "country", "label", "query_group", "prob"]
        deltas.append(df.loc[~df["is_control"], cols])
        controls.append(df.loc[df["is_control"], ["s1_id", "cand_id", "prob"]])
        log(f"{country}: {json.dumps(stats[country])}")
        del todo[country][1][:], blk, df
    delta = pd.concat(deltas, ignore_index=True)
    pe = pd.read_parquet(PRED_EVAL)
    check_controls(pd.concat(controls, ignore_index=True).assign(is_control=True),
                   pd.Series(pe["prob"].to_numpy(), index=pd.MultiIndex.from_arrays(
                       [id_key(pe["s1_id"].to_numpy(object)), id_key(pe["cand_id"].to_numpy(object))])))
    res = {}
    for name in ("pass7", "chainfix"):
        if name == "chainfix":
            pe = pd.concat([pe[~pe["cand_id"].isin(set(delta["cand_id"]))], delta], ignore_index=True)
        ev = Eval(pe, b_ids, {s: truth[s] for s in b_ids}, s1c)
        res[name] = {str(t): {k: v for k, v in ev.score(t).items()
                              if k.startswith("f05") or k in ("precision_corrected", "recall", "singleton_accuracy")}
                     for t in (0.80, 0.85, 0.90)}
        del ev
        log(f"{name}: " + json.dumps(res[name]["0.8"]))
    out = {"stats": stats, "eval": res, "runtime_min": round((time.perf_counter() - t0) / 60, 1),
           "peak_mb": round(peak_rss_mb())}
    (CACHE / "chainfix_eval.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out))


# ---------- test ----------

def run_test(country: str) -> None:
    t0 = time.perf_counter()
    pred_parts = sorted((CACHE / "pred_test").glob(f"{country}-*.parquet"))
    pred = pd.concat([pd.read_parquet(p) for p in pred_parts], ignore_index=True)
    paths = [p for p in part_paths("test") if part_country(p) == country]
    check_block_cols("test", country, paths)
    s1 = pq.read_table(norm_path("test", "source1"), columns=["entity_id"], filters=[("country_norm", "=", country)])
    s1_ids = s1["entity_id"].to_numpy(zero_copy_only=False)
    targets = s1_ids[~np.isin(id_key(s1["entity_id"]), pred["s1"].to_numpy())].tolist()
    pairs = new_pairs("test", country, targets, key_id(np.unique(pred["rec"].to_numpy())).to_pylist())
    blk = block_cols("test", country, pairs).assign(country=country, weight=np.float32(1.0))
    df = rescore("test", country, blk)
    check_controls(df, pred.set_index(["s1", "rec"])["prob"])
    d = df[~df["is_control"]]
    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    out = DELTA_DIR / f"{country}.parquet"
    pd.DataFrame({"s1": id_key(d["s1_id"].to_numpy(object)), "rec": id_key(d["cand_id"].to_numpy(object)),
                  "prob": d["prob"].to_numpy(np.float32), "is_new": d["is_new"].to_numpy()}).to_parquet(out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    stats = {"country": country, "target_s1": len(targets), "target_s1_with_new_pair": int(blk["s1_id"].nunique()),
             "new_pairs": len(blk), "new_pairs_p>=0.8": int((d["is_new"] & (d["prob"] >= 0.8)).sum()),
             "records_rescored": int(d["cand_id"].nunique()), "delta_rows": len(d),
             "runtime_min": round((time.perf_counter() - t0) / 60, 1), "peak_mb": round(peak_rss_mb())}
    (DELTA_DIR / f"{country}.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["eval", "test"])
    ap.add_argument("country", nargs="?", choices=["france", "india", "us"])
    args = ap.parse_args()
    start_memory_guard()
    run_eval() if args.kind == "eval" else run_test(args.country)


if __name__ == "__main__":
    main()
