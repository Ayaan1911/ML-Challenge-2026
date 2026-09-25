"""Load the competition TSVs. First load of each file writes a parquet cache under data/cache/."""
import csv

import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
CACHE = ROOT / "data" / "cache"


def tsv_path(split: str, name: str) -> Path:
    return DATASET / split / f"{split}_{name}.tsv"


def read_tsv(path: Path) -> pd.DataFrame:
    """Every column as a pyarrow-backed string; empty cells stay ''; quotes are literal text."""
    cached = CACHE / f"{path.stem}.parquet"
    if cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cached)
    df = pd.read_csv(path, sep="\t", dtype="string[pyarrow]", keep_default_na=False,
                     na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8")
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached, index=False)
    return df


def load_sources(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split is 'train' or 'test'. Returns (s1, s2, s3)."""
    return tuple(read_tsv(tsv_path(split, f"source{i}")) for i in (1, 2, 3))


def load_truth_df() -> pd.DataFrame:
    return read_tsv(tsv_path("train", "ground_truth"))


def load_truth() -> dict[str, set[str]]:
    """s1_id -> set of matched S2/S3 ids (empty set for singletons)."""
    gt = load_truth_df()
    return {s1: set(filter(None, m.split(","))) for s1, m in
            zip(gt["source1_entity_id"].tolist(), gt["matched_entity_ids"].tolist())}


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n


if __name__ == "__main__":
    # Integrity check: parsed rows must equal line count minus header, with 4 (or 2) clean columns.
    files = [tsv_path(s, n) for s in ("train", "test") for n in ("source1", "source2", "source3")]
    files.append(tsv_path("train", "ground_truth"))
    for p in files:
        df = read_tsv(p)
        expected = _count_lines(p) - 1
        short = int(df.isna().any(axis=1).sum())  # rows with missing trailing fields
        ids = df.iloc[:, 0]
        bad_id = int((~ids.str.match(r"^S[123]-\S+$")).sum())
        dup = int(ids.duplicated().sum())
        ok = len(df) == expected and not (short or bad_id or dup)
        print(f"{p.name:26s} rows={len(df):>9,d} expected={expected:>9,d} short_rows={short} "
              f"bad_ids={bad_id} dup_ids={dup} dtypes={sorted(set(map(str, df.dtypes)))} "
              f"{'OK' if ok else 'PROBLEM'}")
