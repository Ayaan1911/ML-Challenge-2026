"""Pass 11: additive candidate channels on top of chainfix (variant chainfix2). No re-blocking, no retraining.

Channels, within country, added to the Pass 7 candidates plus the chainfix pairs (src/chainfix.py):
  a  extended chain join: every S1 whose space-free name_core key exceeds the blocking cap (50 S1) gets the records
     with identical name_core and identical first house number, uncapped
  b  name + street: identical name_core and identical street signature (addr_core words without digits, sorted,
     deduplicated), skipping keys held by more than 200 S1
Blocking columns come from chainfix.block_cols (the blocking code itself). The new pairs can point at S1 that already
have candidates, which changes those S1's context features, so every record with a row at such an S1 is re-scored as
well (chainfix.rescore(s1_context=True)); untouched control records must reproduce their saved probabilities exactly.

Run: python -m src.chainfix2 eval india|us      -> data/cache/chainfix2_eval/ (one process per country)
     python -m src.chainfix2 eval-score          -> data/cache/chainfix2_eval.json (chainfix + a, chainfix + a + b)
     python -m src.chainfix2 test <country> ab   -> data/cache/pred_test_chainfix2/<country>.parquet (delta only)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking import CONFIG
from src.chainfix import block_cols, check_controls, new_pairs, rescore
from src.data import CACHE, id_key, key_id, load_truth, peak_rss_mb, start_memory_guard
from src.norm_cache import norm_path
from src.split import SPLITS

NAME_CAP = CONFIG["block_cap"]  # the cap that starves chain names in blocking
STREET_CAP = 200
DELTA2_DIR = CACHE / "pred_test_chainfix2"
COMBOS = {"a": ("a",), "ab": ("a", "b")}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


def street_sig(addr: str) -> str:
    return " ".join(sorted({w for w in addr.split() if not any(c.isdigit() for c in w)}))


def load_side(split: str, sources: tuple, country: str, keep: set | None = None) -> pd.DataFrame:
    frames = []
    for s in sources:
        t = pq.read_table(norm_path(split, s), columns=["entity_id", "name_core", "addr_core", "numbers"],
                          filters=[("country_norm", "=", country)]).to_pandas()
        if keep is not None:
            t = t[t["entity_id"].isin(keep)]
        t["num"] = [n[0] if len(n) else "" for n in t.pop("numbers")]
        t["sig"] = [street_sig(a) for a in t.pop("addr_core")]
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def channel_pairs(split: str, country: str, pool: set) -> pd.DataFrame:
    """(s1_id, cand_id, channel) for channels a and b; a pair found by both is labelled a."""
    s1 = load_side(split, ("source1",), country)
    s1 = s1[s1["name_core"] != ""]
    capped = s1.groupby(s1["name_core"].str.replace(" ", "", regex=False))["entity_id"].transform("size") > NAME_CAP
    rec = load_side(split, ("source2", "source3"), country, pool)
    rec = rec[rec["name_core"] != ""]
    a = s1[capped & (s1["num"] != "")].merge(rec[rec["num"] != ""], on=["name_core", "num"], suffixes=("_s1", "_rec"))
    kb = s1[s1["sig"] != ""]
    kb = kb[kb.groupby(["name_core", "sig"])["entity_id"].transform("size") <= STREET_CAP]
    b = kb.merge(rec[rec["sig"] != ""], on=["name_core", "sig"], suffixes=("_s1", "_rec"))
    out = pd.concat([pd.DataFrame({"s1_id": d["entity_id_s1"].to_numpy(object), "cand_id": d["entity_id_rec"].to_numpy(object),
                                   "channel": ch}) for ch, d in (("a", a), ("b", b))], ignore_index=True)
    return out.drop_duplicates(["s1_id", "cand_id"], keep="first")


def drop_existing(pairs: pd.DataFrame, s1k: np.ndarray, rk: np.ndarray) -> pd.DataFrame:
    """Remove pairs that are already candidates (s1k, rk: existing pair keys of the country)."""
    new_rk = id_key(pairs["cand_id"].to_numpy(object))
    m = np.isin(rk, np.unique(new_rk))
    ex = pd.DataFrame({"s": s1k[m], "r": rk[m]}).assign(ex=True)
    got = pd.DataFrame({"s": id_key(pairs["s1_id"].to_numpy(object)), "r": new_rk}).merge(ex, on=["s", "r"], how="left")
    return pairs[got["ex"].isna().to_numpy()].reset_index(drop=True)


# ---------- eval side ----------

EVAL_DIR = CACHE / "chainfix2_eval"


def eval_setup():
    b_ids = (SPLITS / "valsplit_B_s1_ids.txt").read_text(encoding="utf-8").split()
    truth = load_truth()
    truth = {s: truth[s] for s in b_ids}
    s1c = pq.read_table(norm_path("train", "source1"), columns=["entity_id", "country_norm"],
                        filters=[("entity_id", "in", b_ids)]).to_pandas().set_index("entity_id")["country_norm"]
    return b_ids, truth, s1c


def run_eval_country(country: str) -> None:
    """One process per country (memory): new pairs, blocking columns, re-scoring per combo -> EVAL_DIR."""
    from src.postprocess import PRED_EVAL
    t0 = time.perf_counter()
    b_ids, truth, s1c = eval_setup()
    c = pq.read_table(PRED_EVAL, columns=["s1_id", "cand_id", "query_group"], filters=[("country", "=", country)]).to_pandas()
    targets = sorted(set(s1c.index[s1c == country]) - set(c["s1_id"]))
    pool = set(c["cand_id"].unique())
    groups = c.drop_duplicates("cand_id").set_index("cand_id")["query_group"]
    s1k, rk = id_key(c["s1_id"].to_numpy(object)), id_key(c["cand_id"].to_numpy(object))
    del c
    cf = new_pairs("train", country, targets, sorted(pool)).assign(channel="chainfix")
    ch = drop_existing(channel_pairs("train", country, pool), s1k, rk)
    del pool, s1k, rk
    ch = ch[~ch.set_index(["s1_id", "cand_id"]).index.isin(cf.set_index(["s1_id", "cand_id"]).index)]
    allp = pd.concat([cf, ch], ignore_index=True)
    blk = block_cols("train", country, allp[["s1_id", "cand_id"]])
    blk["channel"] = allp["channel"].to_numpy()
    blk["country"] = country
    blk["query_group"] = groups.reindex(blk["cand_id"]).to_numpy(object)
    blk["weight"] = np.where(blk["query_group"] == "other_sampled", 5.0, 1.0).astype(np.float32)
    blk["side"] = "eval"
    blk["label"] = np.array([r in truth.get(s, ()) for s, r in zip(blk["s1_id"], blk["cand_id"])], np.int8)
    inb = blk["s1_id"].isin(set(b_ids))
    stats = {c_: {"pairs": int((blk["channel"] == c_).sum()), "pairs_to_B_s1": int(((blk["channel"] == c_) & inb).sum()),
                  "true": int(blk.loc[blk["channel"] == c_, "label"].sum())} for c_ in ("chainfix", "a", "b")}
    log(f"{country}: {json.dumps(stats)}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for combo, chans in COMBOS.items():
        new = blk[blk["channel"].isin(("chainfix",) + chans)].drop(columns="channel").reset_index(drop=True)
        df = rescore("val", country, new, s1_context=True)
        df = df[(df["side"] == "eval").to_numpy()]
        df[["s1_id", "cand_id", "country", "label", "query_group", "prob", "is_control"]].to_parquet(
            EVAL_DIR / f"{country}_{combo}.parquet")
        stats[f"records_rescored_{combo}"] = int(df.loc[~df["is_control"], "cand_id"].nunique())
    stats["runtime_min"] = round((time.perf_counter() - t0) / 60, 1)
    stats["peak_mb"] = round(peak_rss_mb())
    (EVAL_DIR / f"{country}.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


def run_eval_score() -> None:
    """Control checks, then corrected F0.5 per combo (one prediction table at a time)."""
    from src.postprocess import PRED_EVAL, Eval
    b_ids, truth, s1c = eval_setup()
    b_true = sum(len(v) for v in truth.values())
    sv = pq.read_table(PRED_EVAL, columns=["s1_id", "cand_id", "prob", "label"])
    true_before = int(((sv["label"].to_numpy() == 1) & sv["s1_id"].to_pandas().isin(set(b_ids)).to_numpy()).sum())
    saved = pd.Series(sv["prob"].to_numpy(), index=pd.MultiIndex.from_arrays([id_key(sv["s1_id"]), id_key(sv["cand_id"])]))
    del sv
    parts = {combo: pd.concat([pd.read_parquet(EVAL_DIR / f"{c}_{combo}.parquet") for c in ("india", "us")],
                              ignore_index=True) for combo in COMBOS}
    for combo, d in parts.items():
        check_controls(d[d["is_control"]], saved)
    del saved
    res = {}
    for combo, d in parts.items():
        delta = d[~d["is_control"]].drop(columns="is_control")
        pe = pd.read_parquet(PRED_EVAL)
        pred = pd.concat([pe[~pe["cand_id"].isin(set(delta["cand_id"]))], delta], ignore_index=True)
        del pe
        ev = Eval(pred, b_ids, truth, s1c)
        r = {k: v for k, v in ev.score(0.80).items() if k.startswith("f05") or k in ("precision_corrected", "recall", "singleton_accuracy")}
        r["pair_recall_candidates"] = int(((pred["label"] == 1) & pred["s1_id"].isin(set(b_ids))).sum()) / b_true
        res[combo] = r
        del ev, pred
        log(f"chainfix + {combo}: {json.dumps(r)}")
    out = {"countries": {c: json.loads((EVAL_DIR / f"{c}.json").read_text()) for c in ("india", "us")}, "eval": res,
           "pair_recall_candidates_pass7": true_before / b_true, "peak_mb": round(peak_rss_mb())}
    (CACHE / "chainfix2_eval.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out))


# ---------- test ----------

def run_test(country: str, combo: str) -> None:
    t0 = time.perf_counter()
    pred = pd.concat([pd.read_parquet(p) for p in sorted((CACHE / "pred_test").glob(f"{country}-*.parquet"))], ignore_index=True)
    s1 = pq.read_table(norm_path("test", "source1"), columns=["entity_id"], filters=[("country_norm", "=", country)])
    s1_ids = s1["entity_id"].to_numpy(zero_copy_only=False)
    targets = s1_ids[~np.isin(id_key(s1["entity_id"]), pred["s1"].to_numpy())].tolist()
    pool = set(key_id(np.unique(pred["rec"].to_numpy())).to_pylist())
    cf = new_pairs("test", country, targets, sorted(pool)).assign(channel="chainfix")
    ch = drop_existing(channel_pairs("test", country, pool), pred["s1"].to_numpy(), pred["rec"].to_numpy())
    ch = ch[ch["channel"].isin(COMBOS[combo])]
    ch = ch[~ch.set_index(["s1_id", "cand_id"]).index.isin(cf.set_index(["s1_id", "cand_id"]).index)]
    del pool
    allp = pd.concat([cf, ch], ignore_index=True)
    blk = block_cols("test", country, allp[["s1_id", "cand_id"]]).assign(country=country, weight=np.float32(1.0))
    df = rescore("test", country, blk, s1_context=True)
    check_controls(df, pred.set_index(["s1", "rec"])["prob"])
    d = df[~df["is_control"]]
    DELTA2_DIR.mkdir(parents=True, exist_ok=True)
    out = DELTA2_DIR / f"{country}.parquet"
    pd.DataFrame({"s1": id_key(d["s1_id"].to_numpy(object)), "rec": id_key(d["cand_id"].to_numpy(object)),
                  "prob": d["prob"].to_numpy(np.float32), "is_new": d["is_new"].to_numpy()}).to_parquet(out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    new = d[d["is_new"]]
    ch_of = allp.set_index(["s1_id", "cand_id"])["channel"]
    chan = ch_of.reindex(pd.MultiIndex.from_arrays([new["s1_id"], new["cand_id"]])).to_numpy()
    stats = {"country": country, "combo": combo,
             "new_pairs": {c_: int((chan == c_).sum()) for c_ in ("chainfix", "a", "b")},
             "new_pairs_p>=0.8": {c_: int(((chan == c_) & (new["prob"].to_numpy() >= 0.8)).sum()) for c_ in ("chainfix", "a", "b")},
             "records_rescored": int(d["cand_id"].nunique()), "delta_rows": len(d),
             "runtime_min": round((time.perf_counter() - t0) / 60, 1), "peak_mb": round(peak_rss_mb())}
    (DELTA2_DIR / f"{country}.json").write_text(json.dumps(stats, indent=1))
    log(json.dumps(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["eval", "eval-score", "test"])
    ap.add_argument("country", nargs="?", choices=["france", "india", "us"])
    ap.add_argument("combo", nargs="?", choices=sorted(COMBOS), default="ab")
    args = ap.parse_args()
    start_memory_guard()
    if args.kind == "eval":
        run_eval_country(args.country)
    elif args.kind == "eval-score":
        run_eval_score()
    else:
        run_test(args.country, args.combo)


if __name__ == "__main__":
    main()
