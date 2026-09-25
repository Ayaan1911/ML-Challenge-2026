import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import re
import pandas as pd
from pathlib import Path
from src.config import (
    TEST_DIR, OUTPUT_DIR, PROJECT_ROOT, 
    S1_ID_COL, MATCHED_COL, CANDIDATE_COL, ID_SEP, DELIM
)

def extract_primary_token(text: str) -> str:
    tokens = [t for t in re.sub(r"[^a-z0-9]", " ", str(text).lower()).split() 
              if len(t) >= 3 and t not in ("the", "and", "inc", "ltd", "llc", "pvt", "co")]
    return tokens[0] if tokens else ""

def run():
    print("=" * 65)
    print("AMAZON ML CHALLENGE 2026 — BASELINE SYSTEM (Test Set)")
    print("=" * 65)

    print("Step 1: Loading test data...")
    s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep=DELIM)
    s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep=DELIM)
    s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep=DELIM)
    targets = pd.concat([s2, s3], ignore_index=True)

    print(f"Loaded {len(s1):,} S1 queries and {len(targets):,} target entities.")
    print("Step 2: Building country-partitioned index...")

    index = {}
    for _, row in targets.iterrows():
        country = str(row.get("country", "")).strip().upper()
        token = extract_primary_token(str(row.get("business_name", "")))
        if token:
            key = (country, token)
            if key not in index:
                index[key] = []
            if len(index[key]) < 25:
                index[key].append(str(row["entity_id"]))

    print(f"Indexing completed with {len(index):,} buckets.")
    print("Step 3: Generating candidate pairs and matched results...")

    matches, candidates = [], []
    for _, row in s1.iterrows():
        s1_id = str(row["entity_id"])
        country = str(row.get("country", "")).strip().upper()
        token = extract_primary_token(str(row.get("business_name", "")))

        cands = index.get((country, token), [])
        cand_str = ID_SEP.join(cands) if cands else ""
        top_match = cands[0] if cands else ""

        candidates.append({S1_ID_COL: s1_id, CANDIDATE_COL: cand_str})
        matches.append({S1_ID_COL: s1_id, MATCHED_COL: top_match})

    print("Step 4: Writing output TSVs...")
    df_match = pd.DataFrame(matches)
    df_cand = pd.DataFrame(candidates)

    match_file = OUTPUT_DIR / "matching_results.tsv"
    cand_file = OUTPUT_DIR / "candidate_pairs.tsv"

    df_match.to_csv(match_file, sep=DELIM, index=False)
    df_cand.to_csv(cand_file, sep=DELIM, index=False)
    print(f"✓ Saved {match_file} ({len(df_match):,} rows)")
    print(f"✓ Saved {cand_file} ({len(df_cand):,} rows)")

    validator = PROJECT_ROOT / "utils" / "validate_submission.py"
    if validator.exists():
        print("\nStep 5: Running official submission validator...")
        os.system(f"python3 {validator} -m {match_file} -c {cand_file}")

if __name__ == "__main__":
    run()
