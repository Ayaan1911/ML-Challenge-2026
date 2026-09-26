"""Train/eval split of the PRUNED validation candidate set (Pass 4 fallback: no train sample).

Split by query record, never by pair, so every record keeps all its candidates together:
  - val S1 ids are shuffled (seed 42) into halves A and B;
  - records whose true S1 is in A go to the train side, in B to the eval side;
  - all other queried records (query_group 'other_sampled': train-matched and never-matched) are split
    50/50 at random (seed 42).
Macro F0.5 is evaluated over the half-B S1 ids only.

Writes data/splits/valsplit_{A,B}_s1_ids.txt (committed), data/splits/valsplit_{train,eval}_other_records.txt
(regenerable, gitignored) and data/cache/val_pruned/NNNN.parquet: one file per original blocking part (so each
holds whole records), pruned rows only (keep == True), plus a 'side' column. Run: python -m src.valsplit
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data import CACHE, id_key, key_id, load_truth_df, peak_rss_mb, start_memory_guard
from src.split import SPLITS, load_split

SEED = 42
SRC = CACHE / "cands_val.parquet"
OUT = CACHE / "val_pruned"
OTHER_EVAL_FRAC = 0.5  # eval side gets half of the sampled other records -> false-positive weight 1/(0.2*0.5)


def part_row_groups(f: pq.ParquetFile) -> list[list[int]]:
    """candsets.label() wrote one write_table call per blocking part, so a part is a run of full row groups
    (1,048,576 rows) closed by a shorter one. Checked against the row counts of the part files."""
    full = f.metadata.row_group(0).num_rows
    parts, cur = [], []
    for i in range(f.metadata.num_row_groups):
        cur.append(i)
        if f.metadata.row_group(i).num_rows < full:
            parts.append(cur)
            cur = []
    if cur:
        parts.append(cur)
    want = [pq.ParquetFile(p).metadata.num_rows for p in sorted((CACHE / "cands_val_parts").glob("*.parquet"))]
    got = [sum(f.metadata.row_group(i).num_rows for i in p) for p in parts]
    if got != want:
        raise SystemExit(f"row groups do not line up with cands_val_parts ({len(got)} vs {len(want)} parts)")
    return parts


def halves() -> tuple[np.ndarray, np.ndarray]:
    val = np.array(sorted(load_split("val")), dtype=object)
    perm = np.random.default_rng(SEED).permutation(len(val))
    return np.sort(val[perm[: len(val) // 2]]), np.sort(val[perm[len(val) // 2:]])


def true_s1_in(s1_ids: np.ndarray) -> np.ndarray:
    """Sorted id keys of records whose true S1 is in s1_ids."""
    gt = load_truth_df()
    gt = gt[gt["source1_entity_id"].isin(set(s1_ids)) & (gt["matched_entity_ids"] != "")]
    recs = gt["matched_entity_ids"].str.split(",").explode()
    return np.sort(id_key(recs.astype(str).tolist()))


def isin_sorted(x: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    i = np.searchsorted(sorted_ref, x).clip(max=max(len(sorted_ref) - 1, 0))
    return (sorted_ref[i] == x) if len(sorted_ref) else np.zeros(len(x), bool)


def main() -> None:
    start_memory_guard()
    t0 = time.perf_counter()
    f = pq.ParquetFile(SRC)
    parts = part_row_groups(f)
    a_ids, b_ids = halves()
    a_recs, b_recs = true_s1_in(a_ids), true_s1_in(b_ids)
    others = []  # pass 1: every sampled other record that survives pruning
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=["cand_id", "query_group", "keep"])
        t = t.filter(pc.and_(t["keep"], pc.equal(t["query_group"], "other_sampled")))
        others.append(np.unique(id_key(t["cand_id"])))
    others = np.unique(np.concatenate(others))
    perm = np.random.default_rng(SEED).permutation(len(others))
    other_eval = np.sort(others[perm[: int(len(others) * OTHER_EVAL_FRAC)]])

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, ids in (("A_s1_ids", a_ids), ("B_s1_ids", b_ids)):
        (SPLITS / f"valsplit_{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    for name, keys in (("eval_other_records", other_eval), ("train_other_records", np.setdiff1d(others, other_eval))):
        (SPLITS / f"valsplit_{name}.txt").write_text("\n".join(key_id(keys).to_pylist()) + "\n", encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.parquet"):
        old.unlink()
    stats = {"train": [0, 0, set()], "eval": [0, 0, set()]}  # pairs, positives, records
    for n, groups in enumerate(parts):
        t = f.read_row_groups(groups)
        t = t.filter(t["keep"]).drop_columns(["keep"])
        k = id_key(t["cand_id"])
        other = pc.equal(t["query_group"], "other_sampled").to_numpy(zero_copy_only=False)
        in_a, in_b = isin_sorted(k, a_recs), isin_sorted(k, b_recs)
        if ((in_a | in_b) == other).any():
            raise SystemExit("a target record is neither A nor B, or an other record is val-matched")
        side = np.where(other, np.where(isin_sorted(k, other_eval), "eval", "train"), np.where(in_a, "train", "eval"))
        t = t.append_column("side", pa.array(side))
        tmp = OUT / f"{n:04d}.tmp"
        pq.write_table(t, tmp)
        os.replace(tmp, OUT / f"{n:04d}.parquet")
        lab = t["label"].to_numpy()
        for s in ("train", "eval"):
            m = side == s
            stats[s][0] += int(m.sum())
            stats[s][1] += int(lab[m].sum())
            stats[s][2].update(np.unique(k[m]).tolist())
    summary = {"A_s1": len(a_ids), "B_s1": len(b_ids), "other_records": len(others),
               "other_records_eval": len(other_eval), "parts": len(parts),
               **{f"{s}_{m}": v for s, (p, pos, r) in stats.items()
                  for m, v in (("pairs", p), ("positives", pos), ("records", len(r)))},
               "runtime_s": round(time.perf_counter() - t0, 1), "peak_mb": round(peak_rss_mb())}
    (CACHE / "val_pruned_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
