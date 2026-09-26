"""Pass 6: why the leaderboard (0.78) is far below the eval score (0.93). Distribution check of test vs eval vs
train truth, per country. Reads saved predictions only (data/cache/pred_test/, data/cache/pred_eval.parquet).
Run: python -m src.gap_diagnosis table
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CACHE, id_key, load_truth_df, start_memory_guard
from src.matcher import NormTable
from src.postprocess import THRESHOLD_PATH, best_per_record
from src.split import SPLITS

rng = np.random.default_rng(42)


def _norm_names(split: str, s1: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """name_norm equality for aligned (S1, record) key pairs."""
    a = NormTable(split, ("source1",), s1)
    b = NormTable(split, ("source2", "source3"), rec)
    na = a.table.column("name_norm").to_numpy(zero_copy_only=False)[np.searchsorted(a.keys, s1)]
    nb = b.table.column("name_norm").to_numpy(zero_copy_only=False)[np.searchsorted(b.keys, rec)]
    return na == nb


def profile(s1_all: np.ndarray, s1: np.ndarray, rec: np.ndarray, prob: np.ndarray, t: float, n_records: int,
            split: str) -> dict:
    """s1_all: every S1 scored in this group; (s1, rec, prob): all pruned pairs of the group's queried records."""
    best = best_per_record(rec, prob)
    chosen = best & (prob >= t)
    per_s1 = pd.Series(np.bincount(np.searchsorted(s1_all, s1[chosen]), minlength=len(s1_all)))
    k = per_s1.clip(upper=5).value_counts(normalize=True).reindex(range(6), fill_value=0)
    cp = prob[chosen]
    idx = np.flatnonzero(chosen)
    idx = idx[rng.choice(len(idx), min(50_000, len(idx)), replace=False)]
    return {"S1": len(s1_all), "S1 empty": (per_s1 == 0).mean(), "matches/S1 mean": per_s1.mean(),
            **{f"S1 with {i}{'+' if i == 5 else ''}": k[i] for i in range(6)},
            "cands/record": len(rec) / max(len(np.unique(rec)), 1),
            "records queried": n_records, "records assigned": chosen.sum() / n_records,
            "chosen p<0.9": (cp < 0.9).mean(), "chosen 0.9-0.99": ((cp >= 0.9) & (cp < 0.99)).mean(),
            "chosen p>=0.99": (cp >= 0.99).mean(), "best p<t (records)": (prob[best] < t).mean(),
            "exact name_norm among chosen": _norm_names(split, s1[idx], rec[idx]).mean()}


def truth_profile(country: str) -> dict:
    gt = load_truth_df()
    s1 = pq.read_table(CACHE / "norm_train_source1.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    gt = gt.merge(s1, left_on="source1_entity_id", right_on="entity_id")
    gt = gt[gt["country_norm"] == country]
    n = gt["matched_entity_ids"].map(lambda m: len([x for x in m.split(",") if x]))
    k = n.clip(upper=5).value_counts(normalize=True).reindex(range(6), fill_value=0)
    pairs = gt.assign(r=gt["matched_entity_ids"].str.split(",")).explode("r")
    pairs = pairs[pairs["r"].fillna("") != ""].sample(50_000, random_state=42)
    recs = sum(pq.ParquetFile(CACHE / f"norm_train_{s}.parquet").metadata.num_rows for s in ("source2", "source3"))
    return {"S1": len(gt), "S1 empty": (n == 0).mean(), "matches/S1 mean": n.mean(),
            **{f"S1 with {i}{'+' if i == 5 else ''}": k[i] for i in range(6)},
            "exact name_norm among chosen": _norm_names("train", id_key(pairs["source1_entity_id"].tolist()),
                                                        id_key(pairs["r"].tolist())).mean(),
            "records assigned": n.sum() / max(recs, 1) if False else float("nan")}


def table() -> pd.DataFrame:
    t = json.loads(THRESHOLD_PATH.read_text())["t"]
    cols = {}
    test_s1 = pq.read_table(CACHE / "norm_test_source1.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    test_rec = pd.concat([pq.read_table(CACHE / f"norm_test_{s}.parquet", columns=["country_norm"]).to_pandas()
                          for s in ("source2", "source3")])["country_norm"].value_counts()
    for country in ("france", "india", "us"):
        pr = pa.concat_tables(pq.read_table(p) for p in sorted((CACHE / "pred_test").glob(f"{country}-*.parquet")))
        s1_all = np.sort(id_key(test_s1.loc[test_s1["country_norm"] == country, "entity_id"].tolist()))
        cols[f"test {country}"] = profile(s1_all, pr["s1"].to_numpy(), pr["rec"].to_numpy(), pr["prob"].to_numpy(),
                                          t, int(test_rec[country]), "test")
    ev = pd.read_parquet(CACHE / "pred_eval.parquet")
    b = set((SPLITS / "valsplit_B_s1_ids.txt").read_text().split())
    s1c = pq.read_table(CACHE / "norm_train_source1.parquet", columns=["entity_id", "country_norm"]).to_pandas()
    for country in ("india", "us"):
        e = ev[ev["country"] == country]
        s1_all = np.sort(id_key(s1c.loc[(s1c["country_norm"] == country) & s1c["entity_id"].isin(b), "entity_id"].tolist()))
        s1k, rk = id_key(e["s1_id"].tolist()), id_key(e["cand_id"].tolist())
        inb = np.isin(s1k, s1_all)
        prof = profile(s1_all, s1k, rk, e["prob"].to_numpy(np.float32), t, e["cand_id"].nunique(), "train")
        # S1-side numbers must only count pairs pointing at B S1s; recompute those
        best = best_per_record(rk, e["prob"].to_numpy(np.float32)) & (e["prob"].to_numpy() >= t) & inb
        per_s1 = pd.Series(np.bincount(np.searchsorted(s1_all, s1k[best]), minlength=len(s1_all)))
        k = per_s1.clip(upper=5).value_counts(normalize=True).reindex(range(6), fill_value=0)
        prof.update({"S1 empty": (per_s1 == 0).mean(), "matches/S1 mean": per_s1.mean(),
                     **{f"S1 with {i}{'+' if i == 5 else ''}": k[i] for i in range(6)}})
        cols[f"eval {country}"] = prof
        cols[f"train truth {country}"] = truth_profile(country)
    return pd.DataFrame(cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "parity"])
    args = ap.parse_args()
    start_memory_guard()
    if args.cmd == "parity":
        return parity()
    df = table()
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    df.to_csv(CACHE / "gap_table.csv")



# ---------- end-to-end parity: the test code path on the eval half-B data ----------

PARITY_TSV = CACHE / "parity_matching_results.tsv"


def parity_write() -> None:
    """Same functions as src/run_pipeline.py (iter_features -> predict -> decide -> write_grouped), on val parts,
    keeping eval-side records and half-B S1 rows."""
    from src.matcher import iter_features, load_model, predict
    from src.postprocess import decide
    from src.run_pipeline import write_grouped
    t = json.loads(THRESHOLD_PATH.read_text())["t"]
    booster = load_model()
    s1k, rk, prob = [], [], []
    for p, df, f in iter_features("val"):
        ev = (df["side"] == "eval").to_numpy()
        s1k.append(id_key(df.loc[ev, "s1_id"].to_numpy(object)))
        rk.append(id_key(df.loc[ev, "cand_id"].to_numpy(object)))
        prob.append(predict(booster, f[ev]))
    s1k, rk, prob = np.concatenate(s1k), np.concatenate(rk), np.concatenate(prob)
    sel = decide(rk, prob, t)
    b_all = np.sort(id_key((SPLITS / "valsplit_B_s1_ids.txt").read_text().split()))
    inb = np.isin(s1k, b_all)
    write_grouped(PARITY_TSV, ["source1_entity_id", "matched_entity_ids"], b_all, s1k[sel & inb], rk[sel & inb])
    print(f"wrote {PARITY_TSV.name}: {len(b_all):,} B S1 rows, {int((sel & inb).sum()):,} matched pairs, t={t}")


def independent_score(tsv_path, truth_tsv) -> float:
    """Reads both files from disk with the csv module only; competition F0.5 per S1, averaged over the TSV's rows."""
    import csv
    truth = {}
    with open(truth_tsv, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            truth[row[0]] = {x for x in (row[1] if len(row) > 1 else "").split(",") if x}
    total = n = 0
    with open(tsv_path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            pred = {x for x in (row[1] if len(row) > 1 else "").split(",") if x}
            true = truth[row[0]]
            if not true:
                score = 1.0 if not pred else 0.0
            else:
                tp = len(pred & true)
                score = 0.0 if tp == 0 else 1.25 * (tp / len(pred)) * (tp / len(true)) / (0.25 * tp / len(pred) + tp / len(true))
            total += score
            n += 1
    return total / n


def parity() -> None:
    from src.data import ROOT
    parity_write()
    got = independent_score(PARITY_TSV, ROOT / "dataset" / "train" / "train_ground_truth.tsv")
    want = json.loads(THRESHOLD_PATH.read_text())["f05"]
    print(f"independent F0.5 from the TSV on disk: {got:.6f}; sweep F0.5 at the same t: {want:.6f}; "
          f"difference {got - want:+.2e}")


if __name__ == "__main__":
    main()
