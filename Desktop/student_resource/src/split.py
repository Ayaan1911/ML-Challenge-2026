import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import TRAIN_DIR, DATA_SPLITS_DIR, SEED, S1_ID_COL

def run():
    print("Loading raw training data...")
    s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")

    print(f"Total S1 queries: {len(s1):,} | Total GT: {len(gt):,}")
    print(f"Performing 80/20 train/validation split (seed={SEED})...")

    train_s1 = s1.sample(frac=0.8, random_state=SEED).reset_index(drop=True)
    val_s1 = s1.drop(train_s1.index).reset_index(drop=True)

    train_ids = set(train_s1["entity_id"])
    train_gt = gt[gt[S1_ID_COL].isin(train_ids)].reset_index(drop=True)
    val_gt = gt[~gt[S1_ID_COL].isin(train_ids)].reset_index(drop=True)

    print("Saving parquet files to data_splits/ ...")
    train_s1.to_parquet(DATA_SPLITS_DIR / "train_s1.parquet", index=False)
    val_s1.to_parquet(DATA_SPLITS_DIR / "val_s1.parquet", index=False)
    train_gt.to_parquet(DATA_SPLITS_DIR / "train_gt.parquet", index=False)
    val_gt.to_parquet(DATA_SPLITS_DIR / "val_gt.parquet", index=False)
    print(f"✓ Successfully generated 80/20 split: {len(train_s1):,} train | {len(val_s1):,} val")

if __name__ == "__main__":
    run()
