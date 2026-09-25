"""Small dev slice for fast blocking experiments. Run: python -m src.dev_slice

dev S1   = 10% of validation S1 ids (seed 42)
dev pool = every S2/S3 record matched to a dev S1 + a 10% sample (seed 42) of all other train S2/S3.
The pool is LESS DENSE than real data: 44k S1 compete for each record instead of 1.7-2.2M, so
blocking recall is optimistic, and candidates per S1 are inflated (26 records per S1 instead of ~5)."""
import numpy as np

from src.data import load_sources, load_truth_df
from src.split import SEED, SPLITS, load_split


def load_dev(source: str) -> set[str]:
    """source in {'s1', 's2', 's3'}."""
    return set((SPLITS / f"dev_{source}_ids.txt").read_text(encoding="utf-8").split())


def main() -> None:
    val = sorted(load_split("val"))
    rng = np.random.default_rng(SEED)
    dev_s1 = set(rng.choice(val, size=len(val) // 10, replace=False).tolist())
    gt = load_truth_df()
    gt = gt[gt["source1_entity_id"].isin(dev_s1)]
    matched = set(m for row in gt["matched_entity_ids"].tolist() for m in row.split(",") if m)
    _, s2, s3 = load_sources("train")
    out = {"s1": dev_s1}
    for name, df in (("s2", s2), ("s3", s3)):
        ids = df["entity_id"]
        is_m = ids.isin(matched)
        others = ids[~is_m].to_numpy()
        pick = rng.random(len(others)) < 0.10
        out[name] = set(ids[is_m].tolist()) | set(others[pick].tolist())
        print(f"{name}: {is_m.sum():,} matched to dev S1 + {pick.sum():,} sampled others")
    for name, ids in out.items():
        (SPLITS / f"dev_{name}_ids.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    print(f"dev S1: {len(dev_s1):,} (singletons {(gt['matched_entity_ids'] == '').mean():.2%}); "
          f"dev pool: {len(out['s2']) + len(out['s3']):,} records; true pairs: {len(matched):,}")


if __name__ == "__main__":
    main()
