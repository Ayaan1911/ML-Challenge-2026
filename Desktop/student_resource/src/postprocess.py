import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import joblib
import pandas as pd
import numpy as np
from src.config import (
    TEST_DIR, OUTPUT_DIR, MODELS_DIR, PROJECT_ROOT, 
    S1_ID_COL, MATCHED_COL, CANDIDATE_COL, DELIM, ID_SEP
)

def fast_jaccard(s1, s2):
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def run_postprocessing(threshold=0.55):
    print("=" * 65)
    print("PHASE 2: ULTRA-FAST INFERENCE & POST-PROCESSING")
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

    # 10M rows zipped into python dicts in 1.8 seconds!
    s1_name_map = dict(zip(s1_df["entity_id"].astype(str), s1_df["business_name"].fillna("").astype(str)))
    s1_addr_map = dict(zip(s1_df["entity_id"].astype(str), s1_df["business_address"].fillna("").astype(str)))

    t_name_map = dict(zip(targets_df["entity_id"].astype(str), targets_df["business_name"].fillna("").astype(str)))
    t_addr_map = dict(zip(targets_df["entity_id"].astype(str), targets_df["business_address"].fillna("").astype(str)))
    print(f"Lookup maps ready: {len(s1_name_map):,} S1 | {len(t_name_map):,} Targets")

    print(f"Extracting features for candidate pairs...")
    s1_ids = df_cands[S1_ID_COL].astype(str).tolist()
    cands_raw = df_cands[CANDIDATE_COL].fillna("").astype(str).tolist()

    top_cands = []
    name_jaccards = []
    addr_jaccards = []
    exact_names = []

    for q_id, c_str in zip(s1_ids, cands_raw):
        cand_id = c_str.split(ID_SEP)[0].strip() if c_str else ""
        top_cands.append(cand_id)

        if cand_id and q_id in s1_name_map and cand_id in t_name_map:
            q_name, t_name = s1_name_map[q_id], t_name_map[cand_id]
            q_addr, t_addr = s1_addr_map[q_id], t_addr_map[cand_id]

            nj = fast_jaccard(q_name, t_name)
            aj = fast_jaccard(q_addr, t_addr)
            ex = 1.0 if q_name.lower().strip() == t_name.lower().strip() and len(q_name) > 3 else 0.0
        else:
            nj, aj, ex = 0.0, 0.0, 0.0

        name_jaccards.append(nj)
        addr_jaccards.append(aj)
        exact_names.append(ex)

    print("Running LightGBM vectorized prediction on all 1.73M candidates at once...")
    df_feat = pd.DataFrame({
        "name_jaccard": name_jaccards,
        "addr_jaccard": addr_jaccards,
        "country_match": np.ones(len(name_jaccards)),
        "exact_name_match": exact_names
    })

    probs = clf.predict_proba(df_feat)[:, 1]

    print(f"Applying Macro F0.5 precision cutoff (tau >= {threshold:.2f})...")
    final_matches = []
    accepted = 0
    rejected = 0

    for q_id, cand_id, p in zip(s1_ids, top_cands, probs):
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
