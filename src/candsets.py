"""Sampled, labelled candidate sets for training and validation, against the FULL train S1 index.

Blocking is per S2/S3 query record, so querying a subset of records keeps the real competition between
look-alike S1s (all 2.2M train S1 stay in the index); only the set of queried records shrinks.

  trainsample: records whose true S1 is in the train split, sampled at --train-frac, plus never-matched
               records at the same fraction (natural proportion). Val-matched records are excluded so no
               validation S1 appears as a positive in training.
  val:         every record whose true S1 is in the val split, plus a random --val-other-frac (20%) of all
               other records (train-matched and never-matched). Pairs from those sampled records get
               weight 1/frac (5), because 80% of such records, and the false positives they would add to
               val S1s, are not queried.

Output: data/cache/cands_{trainsample,val}.parquet with every unpruned candidate plus label (1 if the
candidate's true S1 == s1_id), keep (True for the rows read-time pruning keeps = the rows the model
scores), query_group and weight columns. Run: python -m src.candsets {trainsample|val} [--train-frac F]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.blocking import CONFIG, prune_mask, prune_score, run, ALL_KEYS, PRUNE
from src.data import CACHE, load_truth_df
from src.norm_cache import norm_path
from src.split import load_split

SEED = 42


def record_truth() -> pd.DataFrame:
    """One row per matched S2/S3 record: cand_id -> true_s1 (EDA: each record has at most one S1)."""
    gt = load_truth_df()
    gt = gt[gt["matched_entity_ids"] != ""]
    t = gt.assign(cand_id=gt["matched_entity_ids"].str.split(",")).explode("cand_id")
    return pd.DataFrame({"cand_id": t["cand_id"].astype(str).to_numpy(),
                         "true_s1": t["source1_entity_id"].astype(str).to_numpy()})


def all_records() -> np.ndarray:
    return np.concatenate([pd.read_parquet(norm_path("train", s), columns=["entity_id"])["entity_id"]
                           .to_numpy(object) for s in ("source2", "source3")])


def queries(kind: str, train_frac: float, val_other_frac: float) -> tuple[set[str], set[str]]:
    """Returns (query ids, ids sampled from 'other' records that get weight 1/val_other_frac)."""
    rng = np.random.default_rng(SEED)
    rt = record_truth()
    ids = all_records()
    true_s1 = pd.Series(rt["true_s1"].to_numpy(), index=rt["cand_id"].to_numpy()).reindex(ids).to_numpy()
    in_val = pd.Series(true_s1).isin(set(load_split("val"))).to_numpy()
    matched = ~pd.isna(true_s1)
    if kind == "trainsample":
        pool = (matched & ~in_val) | ~matched
        pick = pool & (rng.random(len(ids)) < train_frac)
        return set(ids[pick]), set()
    other = ~in_val & (rng.random(len(ids)) < val_other_frac)
    return set(ids[in_val]) | set(ids[other]), set(ids[other])


def label(name: str, weighted: set[str], weight: float) -> dict:
    """Label the blocking parts one file at a time and stream them into data/cache/{name}.parquet."""
    truth = record_truth()
    stats = {"pairs": 0, "positives": 0, "s1": set()}
    with _Writer(CACHE / f"{name}.parquet") as w:
        for part in sorted((CACHE / f"{name}_parts").glob("*.parquet")):
            c = pd.read_parquet(part).merge(truth, on="cand_id", how="left")
            c["label"] = (c["true_s1"] == c["s1_id"]).astype(np.int8)
            rec = pd.factorize(c["cand_id"])[0]
            c["keep"] = prune_mask(rec, (c[ALL_KEYS].to_numpy() > 0).any(axis=1), prune_score(c),
                                   PRUNE["n"], PRUNE["floor"])  # the rows the model will score
            other = c["cand_id"].isin(weighted)
            c["query_group"] = np.where(other, "other_sampled", "target")
            c["weight"] = np.where(other, weight, 1.0).astype(np.float32)
            w.write(c.drop(columns="true_s1"))
            stats["pairs"] += len(c)
            stats["positives"] += int(c["label"].sum())
            stats["kept"] = stats.get("kept", 0) + int(c["keep"].sum())
            stats["kept_positives"] = stats.get("kept_positives", 0) + int(c.loc[c["keep"], "label"].sum())
            stats["s1"].update(c["s1_id"].unique().tolist())
    return {"pairs": stats["pairs"], "positives": stats["positives"],
            "positive_rate": stats["positives"] / max(stats["pairs"], 1), "s1_with_cands": len(stats["s1"]),
            "kept_pairs": stats.get("kept", 0), "kept_positives": stats.get("kept_positives", 0),
            "kept_positive_rate": stats.get("kept_positives", 0) / max(stats.get("kept", 0), 1)}


class _Writer:
    """Parquet writer that takes the schema from the first frame and publishes the file atomically."""

    def __init__(self, path):
        self.path, self.tmp, self.w = path, path.with_suffix(".tmp"), None

    def __enter__(self):
        return self

    def write(self, df: pd.DataFrame) -> None:
        t = pa.Table.from_pandas(df, preserve_index=False)
        if self.w is None:
            self.w = pq.ParquetWriter(self.tmp, t.schema)
        self.w.write_table(t.cast(self.w.schema))

    def __exit__(self, *exc):
        if self.w is not None:
            self.w.close()
            if exc[0] is None:
                os.replace(self.tmp, self.path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["trainsample", "val"])
    ap.add_argument("--train-frac", type=float, default=0.25)
    ap.add_argument("--val-other-frac", type=float, default=0.20)
    args = ap.parse_args()
    q, weighted = queries(args.kind, args.train_frac, args.val_other_frac)
    name = f"cands_{args.kind}"
    stats = run("train", dict(CONFIG), f"{name}_parts", queries=q)
    summary = {"queries": len(q), **label(name, weighted, 1 / args.val_other_frac),
               "runtime_s": stats["runtime_s"], "peak_mb": stats["peak_mb"]}
    print(json.dumps(summary), flush=True)
    (CACHE / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
