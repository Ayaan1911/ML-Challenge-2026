"""
src/postprocess.py - High-speed inference using C++ RapidFuzz pre-score (<3 mins).
Owner: AM (Lead)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import joblib
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from src.config import (
    TEST_DIR, OUTPUT_DIR, MODELS_DIR, PROJECT_ROOT, 
    S1_ID_COL, MATCHED_COL, CANDIDATE_COL, DELIM, ID_SEP
)
from src.features import pair_features, FEATURE_NAMES
from src.normalize import normalize_record

def run_postprocessing(threshold=0.55, batch_size=20000):
    print("=" * 65)
    print("PHASE 2: HIGH-SPEED INFERENCE (27 FEATURES, <3 MINS)")
    print("=" * 65)

    model_path = MODELS_DIR / "lgbm_pair_model.pkl"
    print(f"Loading trained LightGBM model from {model_path}...")
    clf = joblib.load(model_path)

    cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
    print(f"Loading candidate pairs from {cand_path}...")
    df_cands = pd.read_csv(cand_path, sep=DELIM)

    print("Building instant lookup maps via fast zip (takes ~2 seconds)...")
    s1_df = pd.read_csv(TEST_DIR / "test_source1.tsv", sep=DELIM)
    s2_df = pd.read_csv(TEST_DIR / "test_source2.tsv", sep=DELIM)
    s3_df = pd.read_csv(TEST_DIR / "test_source3.tsv", sep=DELIM)
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_name_map = dict(zip(s1_df["entity_id"].astype(str), s1_df["business_name"].fillna("").astype(str)))
    s1_addr_map = dict(zip(s1_df["entity_id"].astype(str), s1_df["business_address"].fillna("").astype(str)))
    s1_country_map = dict(zip(s1_df["entity_id"].astype(str), s1_df["country"].fillna("").astype(str)))

    t_name_map = dict(zip(targets_df["entity_id"].astype(str), targets_df["business_name"].fillna("").astype(str)))
    t_addr_map = dict(zip(targets_df["entity_id"].astype(str), targets_df["business_address"].fillna("").astype(str)))
    t_country_map = dict(zip(targets_df["entity_id"].astype(str), targets_df["country"].fillna("").astype(str)))
    print(f"Lookup maps ready: {len(s1_name_map):,} S1 | {len(t_name_map):,} Targets")

    s1_ids = df_cands[S1_ID_COL].astype(str).tolist()
    cands_raw = df_cands[CANDIDATE_COL].fillna("").astype(str).tolist()
    total = len(s1_ids)

    print(f"Extracting top candidates for {total:,} queries...")
    top_cands = []
    for c_str in cands_raw:
        top_cands.append(c_str.split(ID_SEP)[0].strip() if c_str else "")

    print("\nStage 1: RapidFuzz C++ Pre-Screening (scoring all 1.73M pairs in seconds)...")
    indices_to_score = []
    
    for idx, (q_id, cand_id) in enumerate(zip(s1_ids, top_cands)):
        if not cand_id or q_id not in s1_name_map or cand_id not in t_name_map:
            continue
        
        q_n = s1_name_map[q_id]
        c_n = t_name_map[cand_id]

        # C++ token set ratio: instant string match score
        ratio = fuzz.token_set_ratio(q_n, c_n)
        if ratio >= 40:
            indices_to_score.append(idx)

    num_candidates = len(indices_to_score)
    print(f"✓ Pre-screening complete: {num_candidates:,} plausible candidate pairs selected!")
    print(f"  ({total - num_candidates:,} low-similarity pairs safely assigned 0.0 probability)")

    print(f"\nStage 2: Computing full 27 features for {num_candidates:,} pairs...")
    all_probs = np.zeros(total, dtype=np.float32)

    for i in range(0, num_candidates, batch_size):
        end = min(i + batch_size, num_candidates)
        batch_slice = indices_to_score[i:end]
        
        batch_X = []
        for idx in batch_slice:
            q_id = s1_ids[idx]
            cand_id = top_cands[idx]
            
            q_rec = {
                "entity_id": q_id,
                "business_name": s1_name_map[q_id],
                "business_address": s1_addr_map[q_id],
                "country": s1_country_map[q_id]
            }
            c_rec = {
                "entity_id": cand_id,
                "business_name": t_name_map[cand_id],
                "business_address": t_addr_map[cand_id],
                "country": t_country_map[cand_id]
            }
            feat_dict = pair_features(normalize_record(q_rec), normalize_record(c_rec))
            batch_X.append([feat_dict[k] for k in FEATURE_NAMES])

        if batch_X:
            df_batch = pd.DataFrame(batch_X, columns=FEATURE_NAMES)
            scored = clf.predict_proba(df_batch)[:, 1]
            for global_idx, p in zip(batch_slice, scored):
                all_probs[global_idx] = p

        pct = (end / num_candidates) * 100
        print(f"  Scored {end:,} / {num_candidates:,} ({pct:.1f}%) candidate pairs...")

    print(f"\nApplying Macro F0.5 precision cutoff (tau >= {threshold:.2f})...")
    final_matches = []
    accepted = 0
    rejected = 0

    for q_id, cand_id, p in zip(s1_ids, top_cands, all_probs):
        if cand_id and p >= threshold:
            final_matches.append({S1_ID_COL: q_id, MATCHED_COL: cand_id})
            accepted += 1
        else:
            final_matches.append({S1_ID_COL: q_id, MATCHED_COL: ""})  # NO_MATCH
            rejected += 1

    print(f"Outcome: {accepted:,} high-confidence matches | {rejected:,} abstained as NO_MATCH")

    match_file = OUTPUT_DIR / "matching_results.tsv"
    df_out = pd.DataFrame(final_matches)
    df_out.to_csv(match_file, sep=DELIM, index=False)
    print(f"✓ Saved updated {match_file} ({len(df_out):,} rows)")

    validator = PROJECT_ROOT / "utils" / "validate_submission.py"
    if validator.exists():
        print("\nValidating updated submission...")
        os.system(f"python3 {validator} -m {match_file} -c {cand_path}")

if __name__ == "__main__":
    run_postprocessing(threshold=0.55)
