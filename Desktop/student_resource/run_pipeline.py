import os
import sys
import subprocess
from pathlib import Path

# Master Pipeline Runner - Amazon ML Challenge 2026
# Executes: Split -> Blocker -> Model Training -> Post-Processing -> Validation -> Packaging

ROOT = Path(__file__).resolve().parent

def step(title, cmd):
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"❌ Error during: {title}")
        sys.exit(1)

def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 — MASTER END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Train/Val Split
    step("Step 1: Checking 80/20 Train/Validation Split", "python3 src/split.py")

    # 2. High-Recall Blocker
    step("Step 2: Running 4-Channel High-Recall Blocker", "python3 src/blocking.py --mode test")

    # 3. Model Training
    step("Step 3: Training LightGBM Pair Classifier", "python3 src/train.py")

    # 4. Precision Post-Processing
    step("Step 4: Vectorized Inference & Precision Filtering", "python3 src/postprocess.py")

    # 5. Official Submission Validation
    step("Step 5: Official Validator Check", "python3 utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv")

    # 6. Package Submission Zip
    step("Step 6: Packaging Submission ZIP", "zip -r submission/submission.zip output/matching_results.tsv output/candidate_pairs.tsv src/ README.md Documentation_template.md")

    print("\n" + "=" * 70)
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY! SUBMISSION READY IN submission/submission.zip")
    print("=" * 70)

if __name__ == "__main__":
    main()
