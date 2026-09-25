"""80/20 split of training S1 ids, seed 42, stratified by (has_match, country). Run once; output is committed."""
from sklearn.model_selection import train_test_split

from src.data import ROOT, load_sources, load_truth_df

SPLITS = ROOT / "data" / "splits"
SEED = 42


def load_split(name: str) -> list[str]:
    """name is 'train' or 'val'."""
    return (SPLITS / f"{name}_s1_ids.txt").read_text(encoding="utf-8").split()


def main() -> None:
    s1 = load_sources("train")[0][["entity_id", "country"]]
    gt = load_truth_df()
    df = s1.merge(gt, left_on="entity_id", right_on="source1_entity_id", how="left", validate="1:1")
    has_match = df["matched_entity_ids"].fillna("").ne("")
    strata = has_match.astype(str) + "|" + df["country"].astype(str)
    counts = strata.map(strata.value_counts())
    strata = strata.where(counts >= 2, "rare")  # a stratum of one would break stratification
    tr, va = train_test_split(df["entity_id"].tolist(), test_size=0.2, random_state=SEED,
                              stratify=strata.tolist())
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", tr), ("val", va)):
        (SPLITS / f"{name}_s1_ids.txt").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    print(f"train={len(tr):,} val={len(va):,}")
    print(strata.groupby(df["entity_id"].isin(set(va)).map({True: "val", False: "train"}))
          .value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
