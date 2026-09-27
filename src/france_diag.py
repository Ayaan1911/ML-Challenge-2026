"""Pass 8b: does France (15% of test S1, absent from eval) explain the leaderboard gap? Saved Pass 7 outputs and
predictions only (data/cache/pred_test/, data/cache/pred_eval.parquet); nothing is re-blocked or re-scored in bulk.

Run: python -m src.france_diag table      -> per-country test vs eval table, printed and saved to data/cache/france_table.csv
     python -m src.france_diag nofrance   -> output/diag_p7_no_france.tsv + .zip (Pass 7 file, France rows emptied)
     python -m src.france_diag samples    -> logs/pass8b_france_samples.txt (30 empty France S1, 30 near-threshold pairs)
"""
import argparse
import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CACHE, ROOT, id_key, key_id, peak_rss_mb, start_memory_guard
from src.matcher import NormTable, featurize, load_model, part_country, part_paths, read_part, s1_context
from src.norm_cache import norm_path
from src.postprocess import best_per_record, load_rules
from src.split import SPLITS

COUNTRIES = ("france", "india", "us")
P_BINS = [0.8, 0.9, 0.95, 0.99, 0.999, 1.0001]
NEG_RATE = 0.5  # training negatives were subsampled at this rate: model odds = true odds / NEG_RATE
P7_TSV = ROOT / "output" / "matching_results_pass7.tsv"
rng = np.random.default_rng(42)


def s1_countries(split: str) -> pd.DataFrame:
    return pq.read_table(CACHE / f"norm_{split}_source1.parquet", columns=["entity_id", "country_norm"]).to_pandas()


def core_equal(split: str, s1: np.ndarray, rec: np.ndarray) -> float:
    a, b = NormTable(split, ("source1",), s1), NormTable(split, ("source2", "source3"), rec)
    ca = a.table.column("name_core").to_numpy(zero_copy_only=False)[np.searchsorted(a.keys, s1)]
    cb = b.table.column("name_core").to_numpy(zero_copy_only=False)[np.searchsorted(b.keys, rec)]
    return float((ca == cb).mean())


def singleton_estimate(s1_idx: np.ndarray, prob: np.ndarray, n_s1: int) -> float:
    """Expected share of S1 with no true match: prod(1 - p) over each S1's pairs, p recalibrated for the
    negative subsampling (true odds = model odds * NEG_RATE). S1 without candidates count as singletons."""
    p = np.clip(prob.astype(np.float64), 0, 1 - 1e-7)
    p = p * NEG_RATE / (1 - p + p * NEG_RATE)
    log_none = np.bincount(s1_idx, weights=np.log1p(-p), minlength=n_s1)
    return float(np.exp(log_none).mean())


def profile(s1_all: np.ndarray, s1: np.ndarray, rec: np.ndarray, prob: np.ndarray, t: float,
            n_records: int, split: str) -> dict:
    """s1_all: sorted S1 keys of the group. Pairs pointing outside s1_all count for the record's best S1 choice
    but not for S1-side numbers (eval: records of half-B S1s also have candidates among other S1s)."""
    best = best_per_record(rec, prob)
    i = np.searchsorted(s1_all, s1).clip(max=len(s1_all) - 1)
    inside = s1_all[i] == s1
    chosen = best & (prob >= t) & inside
    per_s1 = np.bincount(i[chosen], minlength=len(s1_all))
    has_cand = np.bincount(i[inside], minlength=len(s1_all)) > 0
    k = pd.Series(per_s1).clip(upper=5).value_counts(normalize=True).reindex(range(6), fill_value=0)
    cp = prob[chosen]
    hist = np.histogram(cp, P_BINS)[0] / max(len(cp), 1)
    near = best & inside & (prob >= 0.3) & (prob < t)
    idx = np.flatnonzero(chosen)
    idx = idx[rng.choice(len(idx), min(50_000, len(idx)), replace=False)]
    return {"S1": len(s1_all), "S1 no candidates": int((~has_cand).sum()), "S1 no candidates %": (~has_cand).mean(),
            "S1 empty list %": (per_s1 == 0).mean(), "S1 empty, has candidates %": ((per_s1 == 0) & has_cand).mean(),
            "matches/S1 mean": per_s1.mean(),
            **{f"S1 with {j}{'+' if j == 5 else ''}": k[j] for j in range(6)},
            **{f"chosen p {lo:g}-{min(hi, 1):g}": h for lo, hi, h in zip(P_BINS[:-1], P_BINS[1:], hist)},
            "records assigned %": chosen.sum() / n_records,
            "near-miss pairs (best, 0.3<=p<t) per S1": near.sum() / len(s1_all),
            "est. singleton rate": singleton_estimate(i[inside], prob[inside], len(s1_all)),
            "exact name_core among chosen": core_equal(split, s1[idx], rec[idx])}


def table() -> pd.DataFrame:
    t = load_rules()["t"]
    cols = {}
    s1c = s1_countries("test")
    n_rec = pd.concat([pq.read_table(CACHE / f"norm_test_{s}.parquet", columns=["country_norm"]).to_pandas()
                       for s in ("source2", "source3")])["country_norm"].value_counts()
    for c in COUNTRIES:
        pr = pa.concat_tables(pq.read_table(p) for p in sorted((CACHE / "pred_test").glob(f"{c}-*.parquet")))
        s1_all = np.sort(id_key(s1c.loc[s1c["country_norm"] == c, "entity_id"].tolist()))
        cols[f"test {c}"] = profile(s1_all, pr["s1"].to_numpy(), pr["rec"].to_numpy(), pr["prob"].to_numpy(), t,
                                    int(n_rec[c]), "test")
        del pr
        print(f"test {c} done, peak {peak_rss_mb():,.0f} MB", flush=True)
    ev = pq.read_table(CACHE / "pred_eval.parquet", columns=["s1_id", "cand_id", "prob"])
    s1k, rk, prob = id_key(ev["s1_id"]), id_key(ev["cand_id"]), ev["prob"].to_numpy()
    del ev
    b = set((SPLITS / "valsplit_B_s1_ids.txt").read_text(encoding="utf-8").split())
    s1t = s1_countries("train")
    th = json.loads((ROOT / "models" / "threshold.json").read_text())
    for c in ("india", "us"):
        s1_all = np.sort(id_key(s1t.loc[(s1t["country_norm"] == c) & s1t["entity_id"].isin(b), "entity_id"].tolist()))
        # every eval record whose best candidate could be one of this country's B S1s: records are per country
        rec_c = np.unique(rk[np.isin(s1k, s1_all)])
        m = np.isin(rk, rec_c)
        cols[f"eval {c}"] = profile(s1_all, s1k[m], rk[m], prob[m], t, len(rec_c), "train")
        cols[f"eval {c}"]["eval F0.5 (corrected)"] = th[f"f05_corrected[{c}]"]
        print(f"eval {c} done, peak {peak_rss_mb():,.0f} MB", flush=True)
    df = pd.DataFrame(cols)
    df.to_csv(CACHE / "france_table.csv")
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    return df


def nofrance() -> None:
    """Pass 7 matching file with every France S1 row emptied, plus the upload zip."""
    s1c = s1_countries("test")
    france = set(s1c.loc[s1c["country_norm"] == "france", "entity_id"])
    out = ROOT / "output" / "diag_p7_no_france.tsv"
    n = emptied = 0
    with open(P7_TSV, encoding="utf-8", newline="") as f, open(out.with_suffix(".tmp"), "w", encoding="utf-8",
                                                                  newline="") as g:
        g.write(f.readline())
        for line in f:
            s1 = line.split("\t", 1)[0]
            n += 1
            if s1 in france:
                emptied += line.rstrip("\n").split("\t", 1)[1] != ""
                line = f"{s1}\t\n"
            g.write(line)
    out.with_suffix(".tmp").replace(out)
    zpath = ROOT / "output" / "diag_p7_no_france.zip"
    with zipfile.ZipFile(zpath.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, "matching_results.tsv")
    zpath.with_suffix(".tmp").replace(zpath)
    print(f"{out.name}: {n:,} rows, {len(france):,} France S1 rows emptied ({emptied:,} had matches); {zpath.name} "
          f"{zpath.stat().st_size / 2**20:.1f} MB", flush=True)


# ---------- samples ----------

def raw_rows(ids: list[str]) -> pd.DataFrame:
    return pd.concat([pq.read_table(CACHE / f"test_{s}.parquet", filters=[("entity_id", "in", ids)]).to_pandas()
                      for s in ("source1", "source2", "source3")]).set_index("entity_id")


def samples(n: int = 30) -> None:
    t = load_rules()["t"]
    pr = pa.concat_tables(pq.read_table(p) for p in sorted((CACHE / "pred_test").glob("france-*.parquet")))
    s1k, rk, prob = pr["s1"].to_numpy(), pr["rec"].to_numpy(), pr["prob"].to_numpy()
    del pr
    best = best_per_record(rk, prob)
    matched_s1 = np.unique(s1k[best & (prob >= t)])
    empty = np.setdiff1d(np.unique(s1k), matched_s1)
    pick_s1 = rng.choice(empty, n, replace=False)
    near = np.flatnonzero(best & (prob >= 0.3) & (prob < t))
    pick_pairs = rng.choice(near, n, replace=False)
    want_rec = np.unique(np.r_[rk[np.isin(s1k, pick_s1)], rk[pick_pairs]])
    print(f"France: {len(np.unique(s1k)):,} S1 with candidates, {len(empty):,} of them with an empty list; "
          f"{len(near):,} record-best pairs in [0.3, {t}); featurizing {len(want_rec):,} records", flush=True)

    # rebuild the exact model input for every pair of the wanted records (whole records, as in the test run)
    paths = [p for p in part_paths("test") if part_country(p) == "france"]
    ctx = s1_context("test", paths)
    frames = []
    for p, c in zip(paths, ctx):
        df = read_part("test", p)
        m = np.isin(id_key(df["cand_id"].to_numpy(object)), want_rec)
        if m.any():
            frames.append((df[m].reset_index(drop=True), c[m].reset_index(drop=True)))
    df = pd.concat([f for f, _ in frames], ignore_index=True)
    c = pd.concat([c for _, c in frames], ignore_index=True)
    s1q, rq = id_key(df["s1_id"].to_numpy(object)), id_key(df["cand_id"].to_numpy(object))
    feats = featurize(df, NormTable("test", ("source1",), s1q), NormTable("test", ("source2", "source3"), rq), c)
    booster = load_model()
    x = feats.to_numpy(np.float32)
    p_new = booster.predict(x).astype(np.float32)
    contrib = booster.predict(x, pred_contrib=True)[:, :-1]
    saved = pd.Series(prob, index=pd.MultiIndex.from_arrays([s1k, rk]))
    p_saved = saved.reindex(pd.MultiIndex.from_arrays([s1q, rq])).to_numpy()
    print(f"re-predicted {len(df):,} pairs: max |p - saved p| = {np.nanmax(np.abs(p_new - p_saved)):.2e}", flush=True)

    raw = raw_rows(sorted(set(key_id(s1q).to_pylist()) | set(key_id(rq).to_pylist()) |
                          set(key_id(pick_s1).to_pylist())))
    norm_s1, norm_rec = NormTable("test", ("source1",), np.r_[s1q, pick_s1]), NormTable("test", ("source2", "source3"), rq)
    names = feats.columns.tolist()
    rec_best = pd.Series(p_new).groupby(rq).transform("max").to_numpy()

    def show(i: int, L: list[str]) -> None:
        r = key_id(rq[i:i + 1])[0].as_py()
        nr = norm_rec.frame(rq[i:i + 1]).iloc[0]
        top = np.argsort(contrib[i])[:4]
        L.append(f"    {r}  p={p_new[i]:.3f}{'  (record goes elsewhere, best p=%.3f)' % rec_best[i] if rec_best[i] > p_new[i] else ''}"
                 f"\n      raw : {raw.at[r, 'business_name']} | {raw.at[r, 'business_address']}"
                 f"\n      norm: {nr['name_core']} | {nr['addr_norm']}"
                 f"\n      pushed down by: " + ", ".join(f"{names[j]}={x[i, j]:.2f} ({contrib[i, j]:+.2f})" for j in top))

    def head(s: int, L: list[str]) -> None:
        sid = key_id(np.array([s]))[0].as_py()
        ns = norm_s1.frame(np.array([s])).iloc[0]
        L.append(f"\n{sid}\n  raw : {raw.at[sid, 'business_name']} | {raw.at[sid, 'business_address']}"
                 f"\n  norm: {ns['name_core']} | {ns['addr_norm']}")

    L = [f"# Pass 8b France samples (Pass 7 model, t = {t})", "",
         f"## A. {n} France S1 with an EMPTY list that have candidates (top 3 candidates by p)"]
    for s in pick_s1:
        head(s, L)
        rows = np.flatnonzero(s1q == s)
        for i in rows[np.argsort(-p_new[rows])][:3]:
            show(i, L)
    L += ["", f"## B. {n} France record-best pairs with 0.3 <= p < {t}"]
    for j in pick_pairs:
        i = np.flatnonzero((s1q == s1k[j]) & (rq == rk[j]))[0]
        head(s1k[j], L)
        show(i, L)
    out = ROOT / "logs" / "pass8b_france_samples.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}, peak {peak_rss_mb():,.0f} MB", flush=True)


# ---------- checks behind the sample reading ----------

def eval_near_miss() -> pd.DataFrame:
    """Eval side, record-best pairs: how often each pattern the France samples show is a true match.
    pred_eval.parquet is row-aligned with the eval rows of feats_val (both in sorted part order; checked)."""
    t = load_rules()["t"]
    cols = ["s1_id", "cand_id", "country", "label", "query_group", "house_number_conflict", "addr_token_set"]
    f = pd.concat([pq.read_table(p, columns=cols, filters=[("side", "=", "eval")]).to_pandas()
                   for p in sorted((CACHE / "feats_val").glob("*.parquet"))], ignore_index=True)
    pe = pq.read_table(CACHE / "pred_eval.parquet", columns=["s1_id", "cand_id", "prob"]).to_pandas()
    if not ((f["s1_id"].to_numpy() == pe["s1_id"].to_numpy()).all() and (f["cand_id"].to_numpy() == pe["cand_id"].to_numpy()).all()):
        raise SystemExit("pred_eval is not row-aligned with feats_val")
    f["prob"] = pe["prob"].to_numpy()
    del pe
    f = f[best_per_record(id_key(f["cand_id"].to_numpy(object)), f["prob"].to_numpy())]
    f["band"] = np.where(f["prob"] >= t, f"p>={t}", np.where(f["prob"] >= 0.3, f"0.3<=p<{t}", "p<0.3"))
    f["pattern"] = np.select([f["addr_token_set"] < 0, f["house_number_conflict"] > 0],
                             ["record has no address", "house number conflict"], "other")
    f["w"] = np.where(f["query_group"] == "other_sampled", 10.0, 1.0)  # other records were sampled at 1/10
    g = f.groupby(["country", "band", "pattern"])
    out = pd.DataFrame({"pairs": g.size(), "true-match rate": g["label"].mean(),
                        "true-match rate, other records x10": g.apply(lambda d: (d["label"] * d["w"]).sum() / d["w"].sum())})
    print(out.round(3).to_string(), flush=True)
    return out


def blocking_misses() -> pd.DataFrame:
    """S1 without any candidate: does a same-country S2/S3 record carry the identical name_core?"""
    cols = ["entity_id", "country_norm", "name_core", "addr_norm"]
    first_num = r"(\d+)"
    s1 = pq.read_table(norm_path("test", "source1"), columns=cols).to_pandas()
    s1["num"] = s1.pop("addr_norm").str.extract(first_num, expand=False).fillna("")
    s1["S1 sharing its name_core"] = s1.groupby(["country_norm", "name_core"])["entity_id"].transform("size")
    rec = pd.concat([pq.read_table(norm_path("test", s), columns=cols[1:]).to_pandas() for s in ("source2", "source3")])
    rec["num"] = rec.pop("addr_norm").str.extract(first_num, expand=False).fillna("")
    rec = rec[rec["name_core"] != ""]
    core = rec[["country_norm", "name_core"]].drop_duplicates().assign(core=True)
    core_num = rec[rec["num"] != ""].drop_duplicates().assign(core_num=True)
    del rec
    cand = np.unique(np.concatenate([pq.read_table(p, columns=["s1"])["s1"].to_numpy()
                                     for p in sorted((CACHE / "pred_test").glob("*.parquet"))]))
    s1["has_cand"] = np.isin(id_key(s1["entity_id"].tolist()), cand)
    s1 = s1.merge(core, on=["country_norm", "name_core"], how="left").merge(
        core_num, on=["country_norm", "name_core", "num"], how="left")
    for c in ("core", "core_num"):
        s1[c] = s1[c].fillna(False).astype(bool) & (s1["num"] != "" if c == "core_num" else True)
    out = s1.groupby(["country_norm", "has_cand"]).agg(
        S1=("entity_id", "size"), identical_core_record=("core", "mean"),
        identical_core_and_house_number_record=("core_num", "mean"),
        median_S1_sharing_name_core=("S1 sharing its name_core", "median"),
        S1_sharing_name_core_over_50=("S1 sharing its name_core", lambda x: (x > 50).mean()))
    print(out.round(4).to_string(), flush=True)
    return out


def checks() -> None:
    a, b = eval_near_miss(), blocking_misses()
    a.to_csv(CACHE / "france_eval_near_miss.csv")
    b.to_csv(CACHE / "france_blocking_misses.csv")
    print(f"peak {peak_rss_mb():,.0f} MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "nofrance", "samples", "checks"])
    args = ap.parse_args()
    start_memory_guard()
    {"table": table, "nofrance": nofrance, "samples": samples, "checks": checks}[args.cmd]()


if __name__ == "__main__":
    main()
