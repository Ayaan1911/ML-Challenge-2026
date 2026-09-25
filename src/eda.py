"""EDA facts for task card 6.3 step 2. Prints and writes docs/eda_report.md. Run: python -m src.eda"""
import gc
import sys
import time

import pandas as pd

from src.data import ROOT, load_sources, load_truth_df, peak_rss_mb

REPORT = ROOT / "docs" / "eda_report.md"
_lines: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


def block(obj) -> None:
    out("```")
    out(obj if isinstance(obj, str) else obj.to_string())
    out("```")


def source_stats(split: str, sources) -> None:
    out(f"## {split}: records and countries per source\n")
    for label, df in zip(("S1", "S2", "S3"), sources):
        vc = df["country"].value_counts()
        out(f"- **{split} {label}**: {len(df):,} records; countries: "
            + ", ".join(f"`{k}` {v:,} ({v / len(df):.1%})" for k, v in vc.items()))
    out(f"\n## {split}: empty fields and average lengths per source\n")
    rows, lens = [], []
    for label, df in zip(("S1", "S2", "S3"), sources):
        name, addr = df["business_name"].str.strip(), df["business_address"].str.strip()
        rows.append({"source": label, "empty_name": (name == "").mean(), "empty_address": (addr == "").mean(),
                     "both_empty": ((name == "") & (addr == "")).mean()})
        g = pd.DataFrame({"country": df["country"], "name_len": name.str.len(), "addr_len": addr.str.len()})
        lens.append(g.groupby("country").agg(n=("name_len", "size"), name_len=("name_len", "mean"),
                                             addr_len=("addr_len", "mean")).assign(source=label))
    block(pd.DataFrame(rows).set_index("source").map(lambda v: f"{v:.2%}"))
    out("Average length in characters (after strip, empties included):")
    block(pd.concat(lens).reset_index().set_index(["source", "country"]).round(1))


def truth_facts(s1, s2, s3, gt) -> pd.DataFrame:
    out("## train: ground truth\n")
    out(f"- ground-truth rows: {len(gt):,}; S1 rows: {len(s1):,}; "
        f"S1 ids missing from truth: {(~s1['entity_id'].isin(gt['source1_entity_id'])).sum():,}")
    pairs = gt[gt["matched_entity_ids"] != ""].assign(
        m=lambda d: d["matched_entity_ids"].str.split(",")).explode("m", ignore_index=True)[["source1_entity_id", "m"]]
    pairs["m"] = pairs["m"].astype("string[pyarrow]").str.strip()
    pairs = pairs[pairs["m"] != ""]
    n = pairs.groupby("source1_entity_id").size().reindex(gt["source1_entity_id"], fill_value=0)
    country = s1.set_index("entity_id")["country"].reindex(n.index)
    out(f"- singleton share (no matches): **{(n == 0).mean():.2%}** ({(n == 0).sum():,} of {len(n):,})")
    out(f"- matches per S1: mean {n.mean():.2f}, mean over non-singletons {n[n > 0].mean():.2f}, max {n.max()}")
    bucket = n.clip(upper=10).map(lambda k: "10+" if k == 10 else str(k))
    order = [str(k) for k in range(10)] + ["10+"]
    dist = pd.crosstab(bucket, country, normalize="columns").reindex(order).fillna(0)
    dist["ALL"] = bucket.value_counts(normalize=True).reindex(order).fillna(0)
    out("\nMatches per S1 (share of S1 records, per country):")
    block(dist.map(lambda v: f"{v:.2%}"))

    src = pairs["m"].str.slice(0, 3)
    out("## train: S2 vs S3 split of matches\n")
    out("- matched ids by prefix: " + ", ".join(f"`{k}` {v:,} ({v / len(src):.1%})"
                                               for k, v in src.value_counts().items()))
    per = pd.crosstab(pairs["source1_entity_id"], src)
    has2, has3 = per.get("S2-", 0) > 0, per.get("S3-", 0) > 0
    out(f"- non-singleton S1 with matches only in S2: {(has2 & ~has3).mean():.2%}, only S3: "
        f"{(~has2 & has3).mean():.2%}, both: {(has2 & has3).mean():.2%}")
    del per
    dup_within = pairs.duplicated().sum()
    multi = pairs.drop_duplicates()["m"].duplicated(keep=False)
    out(f"- duplicate id inside one truth row: {dup_within:,}")
    out(f"- **S2/S3 ids appearing in more than one truth row: {pairs.drop_duplicates()['m'][multi].nunique():,}** "
        f"(of {pairs['m'].nunique():,} distinct matched ids)")
    if multi.any():
        block(pairs.drop_duplicates()[multi].sort_values("m").head(10).to_string(index=False))

    other = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")["country"]
    pc = pd.DataFrame({"s1_country": country.reindex(pairs["source1_entity_id"]).to_numpy(),
                       "m_country": other.reindex(pairs["m"]).to_numpy()})
    missing = pc["m_country"].isna().sum()
    out("\n## train: country consistency of matched pairs\n")
    out(f"- matched ids not found in S2/S3 files: {missing:,}")
    pc = pc.dropna()
    same = (pc["s1_country"] == pc["m_country"])
    same_ci = pc["s1_country"].str.strip().str.lower() == pc["m_country"].str.strip().str.lower()
    out(f"- **pairs with identical country string: {same.mean():.4%}** "
        f"({(~same).sum():,} differ; {(~same_ci).sum():,} differ case-insensitively)")
    if (~same).any():
        out("Country combinations among differing pairs:")
        block(pc[~same].value_counts().head(10).to_string())

    matched = set(pairs["m"].tolist())
    out("## train: S2/S3 records never matched to any S1\n")
    for label, df in (("S2", s2), ("S3", s3)):
        never = (~df["entity_id"].isin(matched)).sum()
        out(f"- {label}: {never:,} of {len(df):,} never matched ({never / len(df):.1%})")
    return pairs


def show_groups(s1, s2, s3, pairs, k: int = 10) -> None:
    out("\n## train: 10 random matched groups (seed 42)\n")
    s1i = s1.set_index("entity_id")
    oth = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")
    ids = pairs["source1_entity_id"].drop_duplicates().sample(k, random_state=42)
    lines = []
    for sid in ids:
        r = s1i.loc[sid]
        lines.append(f"{sid:<14} | {r.business_name[:45]:<45} | {r.business_address[:70]:<70} | {r.country}")
        for m in pairs.loc[pairs["source1_entity_id"] == sid, "m"]:
            o = oth.loc[m]
            lines.append(f"  {m:<12} | {o.business_name[:45]:<45} | {o.business_address[:70]:<70} | {o.country}")
        lines.append("")
    block("\n".join(lines))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252; names include Devanagari
    t0 = time.time()
    out("# EDA report (generated by `python -m src.eda`)\n")
    test = load_sources("test")
    source_stats("test", test)
    del test
    gc.collect()
    s1, s2, s3 = load_sources("train")
    gt = load_truth_df()
    source_stats("train", (s1, s2, s3))
    pairs = truth_facts(s1, s2, s3, gt)
    show_groups(s1, s2, s3, pairs)
    out(f"_Runtime {time.time() - t0:.0f} s, peak process memory {peak_rss_mb():,.0f} MB._")
    REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
