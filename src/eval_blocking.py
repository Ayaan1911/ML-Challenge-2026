"""Evaluate blocking: dev-slice variants, pruning sweeps, full-scale calibration, real-density validation.

Run: python -m src.eval_blocking            (writes docs/blocking_report.md)
Each blocking run is a separate process so its runtime and peak memory are measured cleanly.
Dev recall is ~10 points optimistic (the dev pool is far less dense than real data): use dev for relative
comparisons and speed, and the real-density validation section (cands_val) for absolute numbers.
"""
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking import ALL_KEYS, SCORES, prune, prune_mask, prune_score, read_cands
from src.data import CACHE, ROOT, load_truth_df
from src.dev_slice import load_dev
from src.metric import macro_f05
from src.norm_cache import norm_path
from src.split import load_split

REPORT = ROOT / "docs" / "blocking_report.md"
PASS2 = ['addr_mode="trigram"', 'k_addr=5', 'prefix_name=4',
         'keys=["key_name_nospace","key_sorted_tokens","key_postcode_num"]']
NOPRUNE = ["prune_n=null"]
VARIANTS = {  # name -> overrides of src.blocking.CONFIG (all stored unpruned); CHOSEN is CONFIG itself
    "pass2": PASS2 + NOPRUNE,
    "pass2 + postcode keys": PASS2[:3] + NOPRUNE,
    "rw df.001 k5 p4": ["rare_df_frac=0.001", "prefix_name=4"] + NOPRUNE,
    "rw df.002 k5 p4": ["rare_df_frac=0.002", "prefix_name=4"] + NOPRUNE,
    "rw df.001 k10 p4": ["rare_df_frac=0.001", "k_addr=10", "prefix_name=4"] + NOPRUNE,
    "v2 chosen (df.002 k5 p3)": [],
}
CHOSEN = "v2 chosen (df.002 k5 p3)"
CALIBRATE = ["pass2", CHOSEN]
PRUNE_NS = (2, 3, 4, 5, 6)
PER_RECORD_STAGES = ("keys", "name_tfidf", "addr_retriever", "dedup_prune", "write")


def slug(variant: str) -> str:
    return "eval_" + "".join(ch if ch.isalnum() else "_" for ch in variant)


def run_blocking(split: str, name: str, sets: list[str], sample: int = 0) -> dict:
    cmd = [sys.executable, "-m", "src.blocking", split, "--name", name, "--fresh", "--set", *sets]
    if sample:
        cmd += ["--sample-records", str(sample)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode:
        raise RuntimeError(res.stderr[-3000:])
    return json.loads(res.stdout.strip().splitlines()[-1])


def truth_for(s1_ids: set[str]) -> dict[str, set[str]]:
    gt = load_truth_df()
    gt = gt[gt["source1_entity_id"].isin(s1_ids)]
    return {s: set(filter(None, m.split(","))) for s, m in
            zip(gt["source1_entity_id"].tolist(), gt["matched_entity_ids"].tolist())}


def dev_truth() -> tuple[pd.DataFrame, dict[str, set[str]], list[str]]:
    dev_s1 = sorted(load_dev("s1"))
    truth = truth_for(set(dev_s1))
    pairs = pd.DataFrame([(s, m) for s, ms in truth.items() for m in ms], columns=["s1_id", "cand_id"])
    country = pd.read_parquet(norm_path("dev", "source1"), columns=["entity_id", "country_norm"])
    pairs = pairs.merge(country.rename(columns={"entity_id": "s1_id", "country_norm": "truth_country"}),
                        on="s1_id")
    return pairs, truth, dev_s1


def score(c: pd.DataFrame, pairs: pd.DataFrame, truth: dict, dev_s1: list[str], n_pool: int):
    hit = pairs.merge(c, on=["s1_id", "cand_id"], how="left")
    found = hit["tfidf_name_cos"].notna()
    pred: dict[str, set[str]] = {}
    for s, m in zip(hit.loc[found, "s1_id"].tolist(), hit.loc[found, "cand_id"].tolist()):
        pred.setdefault(s, set()).add(m)
    row = {"pair_recall": found.mean()}
    row.update({f"recall_{cty}": g.mean() for cty, g in found.groupby(hit["truth_country"])})
    row.update({"f05_ceiling": macro_f05(pred, truth, dev_s1), "pairs": len(c),
                "per_s1": len(c) / len(dev_s1), "per_record": len(c) / n_pool})
    return row, hit


def retriever_recall(hit: pd.DataFrame) -> dict:
    return {r: (hit[r].fillna(0) > 0).mean() for r in ALL_KEYS + SCORES}


def calibrate(sets: list[str], sample: int) -> dict:
    st = run_blocking("test", "eval_calib", sets, sample)
    shutil.rmtree(CACHE / "eval_calib")
    stages, ratio = st["stages_s"], st["n_rec_all"] / st["n_rec"]
    variable = sum(stages.get(k, 0) for k in PER_RECORD_STAGES)  # scales with the number of query records
    fixed = st["runtime_s"] - variable                           # load, S1 indexes, per-country overhead
    return {"ms_per_record": 1000 * variable / st["n_rec"], "est_test_min": (fixed + variable * ratio) / 60,
            "est_test_pairs": st["pairs"] * ratio, "calib_peak_mb": st["peak_mb"], "stages": stages}


def rankings(c: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"max_cos": np.maximum(c["tfidf_name_cos"], c["tfidf_nameaddr_cos"]).to_numpy(),
            "prune_score": prune_score(c)}


def dev_prune_sweep(c: pd.DataFrame, pairs: pd.DataFrame, truth: dict, dev_s1: list[str],
                    n_pool: int) -> pd.DataFrame:
    rec = pd.factorize(c["cand_id"])[0]
    has_key = (c[ALL_KEYS].to_numpy() > 0).any(axis=1)
    rows = []
    for name, sc in rankings(c).items():
        for n in PRUNE_NS:
            row, _ = score(c[prune_mask(rec, has_key, sc, n, 0.0)], pairs, truth, dev_s1, n_pool)
            rows.append({"ranking": name, "prune_n": n, **row})
    return pd.DataFrame(rows)


def val_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recall at REAL density: val-matched records were queried against all 2.2M train S1. Pruning acts
    per query record over all its candidates (train and val S1); recall counts val S1 pairs only."""
    val = set(load_split("val"))
    truth = truth_for(val)
    n_true = sum(len(v) for v in truth.values())
    c = pd.read_parquet(CACHE / "cands_val.parquet",
                        columns=["s1_id", "cand_id", "country", "label", "keep", *ALL_KEYS, *SCORES])
    lab = c["label"].to_numpy().astype(bool) & c["s1_id"].isin(val).to_numpy()
    rec = pd.factorize(c["cand_id"])[0]
    n_rec = int(rec.max()) + 1
    has_key = (c[ALL_KEYS].to_numpy() > 0).any(axis=1)
    rows = [{"ranking": "none (unpruned)", "prune_n": "-", "pair_recall": lab.sum() / n_true,
             "pairs_per_record": len(c) / n_rec}]
    for name, sc in rankings(c).items():
        for n in PRUNE_NS:
            m = prune_mask(rec, has_key, sc, n, 0.0)
            rows.append({"ranking": name, "prune_n": n, "pair_recall": (lab & m).sum() / n_true,
                         "pairs_per_record": m.sum() / n_rec})
    country = pd.read_parquet(norm_path("train", "source1"), columns=["entity_id", "country_norm"])
    country = country[country["entity_id"].isin(val)].set_index("entity_id")["country_norm"]
    true_by_c = pd.Series({s: len(v) for s, v in truth.items()}).groupby(country).sum()
    summary = []
    for tag, mask in (("unpruned", np.ones(len(c), bool)), ("pruned (read time)", c["keep"].to_numpy())):
        hits = c[lab & mask]
        pred: dict[str, set[str]] = {}
        for s1, m in zip(hits["s1_id"].tolist(), hits["cand_id"].tolist()):
            pred.setdefault(s1, set()).add(m)
        by_c = hits.groupby("country").size()
        summary.append({"candidates": tag, "pair_recall": len(hits) / n_true,
                        **{f"recall_{k}": by_c.get(k, 0) / v for k, v in true_by_c.items()},
                        "f05_ceiling": macro_f05(pred, truth, list(truth)), "pairs_per_record": mask.sum() / n_rec})
    return pd.DataFrame(summary), pd.DataFrame(rows), {"val_s1": len(truth), "true_pairs": n_true}


VAL_BIAS = """## How the sampled validation set biases F0.5, and the correction

Validation queries are every record whose true S1 is a val S1, plus a random 20% of all other records.
Every true pair of a val S1 is therefore candidate-eligible, so recall is measured exactly. False
positives are not: a record that belongs to a train S1 (or to nobody) can still be predicted for a val
S1, and only 20% of those records were queried. The other 80% would have added about four times as
many such false positives, so raw validation precision, and with it F0.5, is **overestimated**; the
effect is largest on singleton val S1s, which lose their free 1.0 on any false positive.

Correction: each pair from a sampled "other" record carries `weight = 5` (= 1 / 0.2) in
`cands_val.parquet`. When scoring, count a false positive from such a record as 5 false positives
(expected-value correction): precision = TP / (TP + FP_target + 5 * FP_other). A singleton val S1
already scores 0 on any false positive, so the correction only changes S1s that have matches; what it
cannot recover is the singletons that the unqueried 80% would have hit, so the corrected score is
still slightly optimistic on singletons. Report raw and corrected F0.5 side by side."""


def full_runs() -> list[str]:
    """Summaries of the full test run and the sampled train/val candidate sets, when they exist."""
    out = ["## Full runs with the chosen config", ""]
    test = CACHE / "cands_test"
    if test.exists():
        n = sum(pq.ParquetFile(p).metadata.num_rows for p in test.glob("*.parquet"))
        kept = sum(len(prune(pd.read_parquet(p, columns=["cand_id", *ALL_KEYS, *SCORES])))
                   for p in test.glob("*.parquet"))
        log = (ROOT / "logs" / "blocking_cands_test.log").read_text(encoding="utf-8").splitlines()
        done = [line for line in log if "done:" in line]
        st = json.loads(done[-1].split("done: ", 1)[1]) if done else {"n_rec": 1}
        out.append(f"- **test**: {st['n_rec']:,} query records, {n:,} stored (unpruned) pairs "
                   f"({n / st['n_rec']:.2f} per record), **{kept:,} pairs after read-time pruning** "
                   f"({kept / st['n_rec']:.2f} per record) = the future candidate_pairs.tsv; runtime "
                   f"{st.get('runtime_s', 0) / 60:.1f} min, peak {st.get('peak_mb', 0):,} MB")
    for kind in ("val", "trainsample"):
        f = CACHE / f"cands_{kind}_summary.json"
        if f.exists():
            d = json.loads(f.read_text())
            out.append(f"- **{kind}**: {d['queries']:,} query records, {d['pairs']:,} pairs "
                       f"({d['positives']:,} positives, rate {d['positive_rate']:.4f}); after pruning "
                       f"{d['kept_pairs']:,} pairs ({d['kept_positives']:,} positives, rate "
                       f"{d['kept_positive_rate']:.4f}); runtime {d['runtime_s'] / 60:.1f} min, "
                       f"peak {d['peak_mb']:,} MB")
    if (CACHE / "cands_val.parquet").exists():
        summary, sweep, meta = val_analysis()
        out += ["", f"### Real-density validation ({meta['val_s1']:,} val S1, {meta['true_pairs']:,} true "
                "pairs; every val-matched record queried against all 2.2M train S1)", "", md_table(summary),
                "", "Pruning sweep at real density (per query record, over all its candidates):", "",
                md_table(sweep)]
    return out


def md_table(df: pd.DataFrame) -> str:
    def fmt(v) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.4f}" if abs(v) <= 1.5 else f"{v:,.1f}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def main() -> None:
    pairs, truth, dev_s1 = dev_truth()
    n_pool = len(load_dev("s2")) + len(load_dev("s3"))
    rows, rec_rows = [], []
    for variant, sets in VARIANTS.items():
        st = run_blocking("dev", slug(variant), sets)
        c = read_cands(slug(variant))
        row, hit = score(c, pairs, truth, dev_s1, n_pool)
        rows.append({"variant": variant, **row, "dev_runtime_s": st["runtime_s"]})
        rec_rows.append({"variant": variant, **retriever_recall(hit)})
        print(json.dumps(rows[-1]), flush=True)
    chosen = read_cands(slug(CHOSEN))
    row, _ = score(prune(chosen), pairs, truth, dev_s1, n_pool)
    rows.append({"variant": CHOSEN + " + read-time prune", **row})
    table = pd.DataFrame(rows)
    for variant in CALIBRATE:
        cal = calibrate(VARIANTS[variant], 50_000)
        i = table.index[table["variant"] == variant][0]
        for k in ("ms_per_record", "est_test_min", "est_test_pairs"):
            table.loc[i, k] = cal[k]
        print(variant, json.dumps(cal), flush=True)
    lines = [
        "# Blocking report (generated by `python -m src.eval_blocking`)", "",
        f"Dev slice: {len(dev_s1):,} S1, {n_pool:,} S2/S3 records, {len(pairs):,} true pairs. **The dev pool is "
        "far less dense than real data** (44k S1 instead of 2.2M compete for each record): dev recall is "
        "~10 points optimistic (compare the real-density validation below) and pairs per S1 are inflated. "
        "Use dev for relative comparisons only. Speed columns: 50,000 random test records against the FULL "
        "test S1 index, per-record stages scaled to 9.97M records.",
        "", "## Dev: variants (stored unpruned) and the chosen config with read-time pruning", "",
        md_table(table), "", "## Dev: recall of each retriever alone", "", md_table(pd.DataFrame(rec_rows)), "",
        f"## Dev: pruning sweep on `{CHOSEN}`", "",
        "Keep every key hit plus the top prune_n S1 per record by the ranking score. max_cos = "
        "max(tfidf_name_cos, tfidf_nameaddr_cos); prune_score = tfidf_name_cos + tfidf_nameaddr_cos + "
        "addr_word_idf / 20.", "", md_table(dev_prune_sweep(chosen, pairs, truth, dev_s1, n_pool)), "",
        *full_runs(), "", VAL_BIAS]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
