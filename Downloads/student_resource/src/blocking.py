import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import argparse
from collections import defaultdict
import pandas as pd
from src.config import (
    DATASET_DIR, OUTPUT_DIR, TOP_K_CANDIDATES, 
    COMMON_BUSINESS_WORDS, S1_ID_COL, CANDIDATE_COL, ID_SEP
)

def get_tokens(name):
    return [w for w in re.findall(r"\b[a-z0-9]{3,}\b", str(name).lower()) if w not in COMMON_BUSINESS_WORDS]

def extract_compressed_prefix(name):
    # Handles Wal-Mart vs Walmart, McDonald vs Mc Donald (up to 8 chars)
    clean = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return clean[:8] if len(clean) >= 5 else ""

def extract_addr_anchors(addr):
    # Extracts street numbers and 5-6 digit postal codes
    text = str(addr).lower()
    nums = re.findall(r"\b\d{1,5}\b", text)
    street_num = nums[0] if nums else ""
    zips = re.findall(r"\b\d{5,6}\b", text)
    zip_code = zips[0] if zips else ""
    return street_num, zip_code

class StateOfTheArtBlocker:
    def __init__(self, top_k=TOP_K_CANDIDATES):
        self.top_k = top_k
        self.bigram_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.compressed_index = defaultdict(list)
        self.addr_index = defaultdict(list)

    def fit(self, targets: pd.DataFrame):
        print(f"Indexing all {len(targets):,} targets with 4-Channel High-Recall Engine...")
        t_ids = targets["entity_id"].astype(str).values
        t_names = targets["business_name"].fillna("").astype(str).values
        t_addrs = targets["business_address"].fillna("").astype(str).values
        t_ctrys = targets["country"].fillna("").astype(str).str.upper().values

        for t_id, name, addr, c in zip(t_ids, t_names, t_addrs, t_ctrys):
            tokens = get_tokens(name)
            comp_prefix = extract_compressed_prefix(name)
            s_num, zip_code = extract_addr_anchors(addr)
            name_prefix = name[:3].lower().strip()

            # Channel 1: Skip-Bigrams (Distance 1 & 2)
            n_tokens = len(tokens)
            for i in range(min(n_tokens, 4)):
                for j in range(i + 1, min(n_tokens, i + 3)):
                    key = (c, tokens[i], tokens[j])
                    if len(self.bigram_index[key]) < 20:
                        self.bigram_index[key].append(t_id)

            # Channel 2: Distinctive single tokens (>= 5 chars)
            for tok in tokens[:3]:
                if len(tok) >= 5 and len(self.token_index[(c, tok)]) < 20:
                    self.token_index[(c, tok)].append(t_id)

            # Channel 3: Compressed Alphanumeric Prefix (e.g. walmart)
            if comp_prefix:
                key = (c, comp_prefix)
                if len(self.compressed_index[key]) < 20:
                    self.compressed_index[key].append(t_id)

            # Channel 4: Address Anchors (Street Num or Zip + Name Prefix)
            if s_num and len(name_prefix) >= 2:
                key = (c, s_num, name_prefix)
                if len(self.addr_index[key]) < 20:
                    self.addr_index[key].append(t_id)
            if zip_code and len(name_prefix) >= 2:
                key = (c, zip_code, name_prefix)
                if len(self.addr_index[key]) < 20:
                    self.addr_index[key].append(t_id)

        print(f"Index complete: {len(self.bigram_index):,} bigrams | {len(self.compressed_index):,} compressed | {len(self.addr_index):,} address anchors.")

    def block_queries(self, queries: pd.DataFrame) -> dict:
        print(f"Blocking {len(queries):,} queries...")
        q_ids = queries["entity_id"].astype(str).values
        q_names = queries["business_name"].fillna("").astype(str).values
        q_addrs = queries["business_address"].fillna("").astype(str).values
        q_ctrys = queries["country"].fillna("").astype(str).str.upper().values

        candidates = {}
        for q_id, name, addr, c in zip(q_ids, q_names, q_addrs, q_ctrys):
            tokens = get_tokens(name)
            comp_prefix = extract_compressed_prefix(name)
            s_num, zip_code = extract_addr_anchors(addr)
            name_prefix = name[:3].lower().strip()

            cand_set = set()

            # Channel 1: Skip-Bigrams
            n_tokens = len(tokens)
            for i in range(min(n_tokens, 4)):
                for j in range(i + 1, min(n_tokens, i + 3)):
                    cand_set.update(self.bigram_index.get((c, tokens[i], tokens[j]), []))

            # Channel 2: Distinctive tokens
            for tok in tokens[:3]:
                if len(tok) >= 5:
                    cand_set.update(self.token_index.get((c, tok), []))

            # Channel 3: Compressed Prefix
            if comp_prefix:
                cand_set.update(self.compressed_index.get((c, comp_prefix), []))

            # Channel 4: Address Anchors
            if s_num and len(name_prefix) >= 2:
                cand_set.update(self.addr_index.get((c, s_num, name_prefix), []))
            if zip_code and len(name_prefix) >= 2:
                cand_set.update(self.addr_index.get((c, zip_code, name_prefix), []))

            candidates[q_id] = list(cand_set)[:self.top_k]

        return candidates

def run(mode="test"):
    dir_path = DATASET_DIR / mode
    prefix = f"{mode}_"
    print(f"Loading {mode} catalogs...")
    df_s1 = pd.read_csv(dir_path / f"{prefix}source1.tsv", sep="\t")
    df_s2 = pd.read_csv(dir_path / f"{prefix}source2.tsv", sep="\t")
    df_s3 = pd.read_csv(dir_path / f"{prefix}source3.tsv", sep="\t")
    targets = pd.concat([df_s2, df_s3], ignore_index=True)

    blocker = StateOfTheArtBlocker(top_k=TOP_K_CANDIDATES)
    blocker.fit(targets)
    cands = blocker.block_queries(df_s1)

    out_rows = [{S1_ID_COL: k, CANDIDATE_COL: ID_SEP.join(v)} for k, v in cands.items()]
    out_df = pd.DataFrame(out_rows)
    out_path = OUTPUT_DIR / "candidate_pairs.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"✓ Saved updated {out_path} ({len(out_df):,} queries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", choices=["train", "test"])
    args = parser.parse_args()
    run(mode=args.mode)
