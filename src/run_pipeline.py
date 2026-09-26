"""End-to-end test run: pruned test candidate parts -> features -> LightGBM -> decision layer -> submission TSVs.

Needs: data/cache/cands_test/ (python -m src.blocking test), models/lgbm.txt (python -m src.matcher train),
models/threshold.json (python -m src.postprocess sweep).
Writes output/matching_results.tsv (one row per test S1, empty list = no match) and output/candidate_pairs.tsv
(the exact pruned candidate set the model scored, grouped per S1; every matched id is also a candidate).
Run: python -m src.run_pipeline test
"""
import argparse
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.data import ROOT, id_key, key_id, peak_rss_mb, start_memory_guard
from src.matcher import iter_features, load_model, log, predict
from src.norm_cache import norm_path
from src.postprocess import THRESHOLD_PATH, decide

OUT = ROOT / "output"
BLOCK = 100_000  # S1 rows per write batch


def write_grouped(path, header: list[str], s1_all: np.ndarray, pair_s1: np.ndarray, pair_rec: np.ndarray) -> None:
    """One row per key of s1_all (sorted output), ids of its pairs comma-joined, '' when it has none."""
    order = np.lexsort((pair_rec, pair_s1))
    s1s, recs = pair_s1[order], pair_rec[order]
    start, end = np.searchsorted(s1s, s1_all, "left"), np.searchsorted(s1s, s1_all, "right")
    opts = pacsv.WriteOptions(include_header=False, delimiter="\t", quoting_style="none")
    schema = pa.schema([(header[0], pa.string()), (header[1], pa.string())])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(("\t".join(header) + "\n").encode())
        with pacsv.CSVWriter(f, schema, write_options=opts) as w:
            for b in range(0, len(s1_all), BLOCK):
                lo, hi = start[b], end[min(b + BLOCK, len(s1_all)) - 1]
                offsets = np.r_[start[b:b + BLOCK], hi] - lo
                lists = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), key_id(recs[lo:hi]))
                w.write_table(pa.table({header[0]: key_id(s1_all[b:b + BLOCK]),
                                        header[1]: pc.binary_join(lists, ",")}, schema=schema))
    tmp.replace(path)


def run_test() -> dict:
    stages, t0 = {}, time.perf_counter()
    t = json.loads(THRESHOLD_PATH.read_text())["t"]
    booster = load_model()
    s1k, rk, prob = [], [], []
    for p, df, f in iter_features("test"):
        s1k.append(id_key(df["s1_id"].to_numpy(object)))
        rk.append(id_key(df["cand_id"].to_numpy(object)))
        prob.append(predict(booster, f))
        log(f"scored {p.name}: {len(df):,} pairs")
    s1k, rk, prob = np.concatenate(s1k), np.concatenate(rk), np.concatenate(prob)
    stages["features_and_scoring"] = {"s": round(time.perf_counter() - t0, 1), "peak_mb": round(peak_rss_mb())}

    t1 = time.perf_counter()
    sel = decide(rk, prob, t)
    s1_all = np.sort(id_key(pq.read_table(norm_path("test", "source1"), columns=["entity_id"])["entity_id"]))
    if not np.isin(np.unique(s1k), s1_all).all():
        raise SystemExit("candidate S1 ids not in test source1")
    stages["decision"] = {"s": round(time.perf_counter() - t1, 1), "peak_mb": round(peak_rss_mb())}

    t2 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    write_grouped(OUT / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"], s1_all, s1k[sel], rk[sel])
    write_grouped(OUT / "candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"], s1_all, s1k, rk)
    stages["write_tsv"] = {"s": round(time.perf_counter() - t2, 1), "peak_mb": round(peak_rss_mb())}

    matched_s1 = np.unique(s1k[sel])
    summary = {"threshold": t, "test_s1": len(s1_all), "candidate_pairs": len(rk), "matched_pairs": int(sel.sum()),
               "s1_with_match": len(matched_s1), "s1_with_candidates": len(np.unique(s1k)),
               "records_matched": int(sel.sum()), "records_scored": len(np.unique(rk)),
               "runtime_min": round((time.perf_counter() - t0) / 60, 1), "peak_mb": round(peak_rss_mb()),
               "stages": stages,
               "matching_results_mb": round((OUT / "matching_results.tsv").stat().st_size / 2**20, 1),
               "candidate_pairs_mb": round((OUT / "candidate_pairs.tsv").stat().st_size / 2**20, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["test"])
    ap.parse_args()
    start_memory_guard()
    run_test()


if __name__ == "__main__":
    main()
