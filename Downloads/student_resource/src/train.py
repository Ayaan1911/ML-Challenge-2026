import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from src.config import (
    DATA_SPLITS_DIR, TRAIN_DIR, PROJECT_ROOT, 
    SEED, COMMON_BUSINESS_WORDS, S1_ID_COL, MATCHED_COL
)
from src.features import pair_features
from src.normalize import normalize_record

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def build_training_pairs(sample_size=30000):
    print(f"Loading {sample_size:,} training queries and ground truth...")
    train_s1 = pd.read_parquet(DATA_SPLITS_DIR / "train_s1.parquet").head(sample_size)
    train_gt = pd.read_parquet(DATA_SPLITS_DIR / "train_gt.parquet")

    gt_map = {}
    for _, row in train_gt.iterrows():
        s1 = str(row[S1_ID_COL])
        m = str(row[MATCHED_COL])
        if pd.notna(m) and m not in ("", "nan", "None"):
            gt_map[s1] = set(re.split(r"[, \t]+", m.strip()))

    print("Loading target catalogs (S2 + S3)...")
    s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
    s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")
    targets = pd.concat([s2, s3], ignore_index=True)
    target_dict = {str(r["entity_id"]): r.to_dict() for _, r in targets.head(500000).iterrows()}

    # Index targets for negative mining
    print("Generating candidate pairs for training...")
    index = defaultdict(list)
    for t_id, r in target_dict.items():
        c = str(r.get("country", "")).strip().upper()
        tokens = [w for w in re.findall(r"\b[a-z0-9]{3,}\b", str(r.get("business_name", "")).lower()) if w not in COMMON_BUSINESS_WORDS]
        for tok in tokens[:2]:
            if len(index[(c, tok)]) < 20:
                index[(c, tok)].append(t_id)

    X_list = []
    y_list = []

    for _, row in train_s1.iterrows():
        s1_id = str(row["entity_id"])
        c = str(row.get("country", "")).strip().upper()
        q_norm = normalize_record(row.to_dict())
        true_ids = gt_map.get(s1_id, set())

        # Retrieve candidates from index
        tokens = [w for w in re.findall(r"\b[a-z0-9]{3,}\b", str(row.get("business_name", "")).lower()) if w not in COMMON_BUSINESS_WORDS]
        cands = set()
        for tok in tokens[:2]:
            cands.update(index.get((c, tok), []))

        # 1. Add Positive Pair
        for t_id in true_ids:
            if t_id in target_dict:
                feats = pair_features(q_norm, normalize_record(target_dict[t_id]))
                X_list.append(feats)
                y_list.append(1)

        # 2. Add Hard Negative Pairs (candidates that are not true matches)
        neg_count = 0
        for cand_id in cands:
            if cand_id not in true_ids and cand_id in target_dict:
                feats = pair_features(q_norm, normalize_record(target_dict[cand_id]))
                X_list.append(feats)
                y_list.append(0)
                neg_count += 1
                if neg_count >= 3:  # 1 positive : 3 negatives
                    break

    df_X = pd.DataFrame(X_list)
    y = np.array(y_list)
    print(f"Training dataset built: {len(df_X):,} pairs | Positives: {(y == 1).sum():,} | Negatives: {(y == 0).sum():,}")
    return df_X, y

def train():
    print("=" * 65)
    print("PHASE 2: TRAINING LIGHTGBM PAIR CLASSIFIER")
    print("=" * 65)
    X, y = build_training_pairs(sample_size=30000)

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1
    )

    print("Fitting LightGBM model...")
    clf.fit(X, y)

    model_path = MODELS_DIR / "lgbm_pair_model.pkl"
    joblib.dump(clf, model_path)
    print(f"✓ Model successfully saved to {model_path}")

    # Print Feature Importances
    print()
    print("Feature Importances:")
    for feat, imp in zip(X.columns, clf.feature_importances_):
        print(f"  {feat:<20}: {imp}")

if __name__ == "__main__":
    train()
