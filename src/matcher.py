"""Pair scoring: features for pruned candidate parts, streamed part by part, and the LightGBM model.

Features per (S1, record) pair:
  - features.pair_features_frame on the two NormRecords (every Table 5.2 column is joined, so swapping in the
    real normalize.py / features.py only means rerunning norm_cache, blocking and the commands below)
  - blocking columns: key flags, TF-IDF cosines, rare-word address score, prune_score
  - record side: candidates of this record, rank of this S1 by each score (1 = best, ties share the best
    rank), gap to the record's best score
  - S1 side: candidate records pointing at this S1 and this record's rank among them by prune_score, both
    weighted by the sampling weight (val 'other' records count 5x, so val matches the fully queried test)
  - source flag (S3 vs S2)
Country is never a feature: France is absent from training.

Run: python -m src.matcher features val   -> data/cache/feats_val/NNNN.parquet
     python -m src.matcher train          -> models/lgbm.txt, docs/feature_importance.md
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.blocking import ALL_KEYS, SCORES, prune, prune_score
from src.data import CACHE, ROOT, id_key, peak_rss_mb, start_memory_guard
from src.features import FEATURE_NAMES, pair_features_frame
from src.norm_cache import SCHEMA as NORM_SCHEMA, norm_path

MODELS = ROOT / "models"
MODEL_PATH = MODELS / "lgbm.txt"
NORM_COLS = NORM_SCHEMA.names
BLOCK_COLS = ALL_KEYS + SCORES + ["prune_score"]
RANK_SCORES = SCORES + ["prune_score", "name_token_set", "addr_token_set"]
CONTEXT_COLS = (["rec_n_cands"] + [f"rec_rank_{c}" for c in RANK_SCORES] + [f"rec_gap_{c}" for c in RANK_SCORES]
                + ["s1_n_cands", "s1_rank", "is_s3"])
MODEL_FEATURES = FEATURE_NAMES + BLOCK_COLS + CONTEXT_COLS
META_COLS = ["s1_id", "cand_id", "country", "label", "weight", "query_group", "side"]
PARAMS = {"objective": "binary", "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "seed": 42}
NORM_GROUP = 8  # parts whose norm records are held at once (Pass 5: a whole country peaked at 4.7 GB)
MAX_ROUNDS = 6000  # Pass 7: Pass 5 hit the old 3,000 cap still improving; early stopping after 100 flat rounds
NEG_RATE = 0.5  # share of negatives kept for training (Pass 5: 0.5 keeps the dataset build near 3 GB, see docs/runbook.md)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


# ---------- candidate parts ----------

def part_paths(kind: str) -> list[Path]:
    """val: pruned split parts from src.valsplit; test: unpruned blocking parts (pruned when read)."""
    d = CACHE / {"val": "val_pruned", "test": "cands_test"}[kind]
    paths = sorted(d.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parts in {d}")
    return paths


def read_part(kind: str, path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if kind == "test":
        df = prune(df).reset_index(drop=True)
        df["weight"] = np.float32(1.0)
    df["prune_score"] = prune_score(df)
    return df


def part_country(path: Path) -> str:
    return pq.read_table(path, columns=["country"]).slice(0, 1)["country"][0].as_py()


# ---------- norm records ----------

def isin_sorted(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if not len(ref):
        return np.zeros(len(x), bool)
    return ref[np.searchsorted(ref, x).clip(max=len(ref) - 1)] == x


class NormTable:
    """Normalized records with the given id keys, held as one Arrow table sorted by key."""

    def __init__(self, split: str, sources: tuple[str, ...], keys: np.ndarray):
        keys = np.unique(keys)
        tables, tkeys = [], []
        for s in sources:
            for b in pq.ParquetFile(norm_path(split, s)).iter_batches(batch_size=500_000, columns=NORM_COLS):
                k = id_key(b["entity_id"])
                m = isin_sorted(k, keys)
                if m.any():
                    tables.append(pa.Table.from_batches([b]).filter(pa.array(m)))
                    tkeys.append(k[m])
        tkeys = np.concatenate(tkeys) if tkeys else np.empty(0, np.int64)
        order = np.argsort(tkeys, kind="stable")
        self.keys = tkeys[order]
        self.table = pa.concat_tables(tables).take(pa.array(order)) if tables else None
        if len(self.keys) != len(keys):
            raise SystemExit(f"{split} {sources}: {len(keys) - len(self.keys)} candidate ids missing from the norm cache")

    def frame(self, keys: np.ndarray) -> pd.DataFrame:
        return self.table.take(pa.array(np.searchsorted(self.keys, keys))).to_pandas()


# ---------- context features ----------

def _grouped_rank(group: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """1 + (weighted) count of rows in the same group with a strictly higher score."""
    order = np.lexsort((-score, group))
    g, s = group[order], score[order]
    w = np.ones(len(g)) if weight is None else weight[order].astype(np.float64)
    before = np.cumsum(w) - w  # weight of all earlier rows (all groups)
    new_group = np.r_[True, g[1:] != g[:-1]]
    new_tie = new_group | np.r_[True, s[1:] != s[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, before, 0))
    tie_start = np.maximum.accumulate(np.where(new_tie, before, 0))
    rank = np.empty(len(g), np.float32)
    rank[order] = 1 + tie_start - group_start
    return rank


def _group_max(group: np.ndarray, score: np.ndarray) -> np.ndarray:
    codes, inv = np.unique(group, return_inverse=True)
    best = np.full(len(codes), -np.inf)
    np.maximum.at(best, inv, score)
    return best[inv]


def s1_context(kind: str, paths: list[Path]) -> list[pd.DataFrame]:
    """S1-side context needs every pair pointing at an S1, across parts: one light pass over all parts."""
    s1, score, weight, sizes = [], [], [], []
    for p in paths:
        df = read_part(kind, p)
        s1.append(id_key(df["s1_id"].to_numpy(object)))
        score.append(df["prune_score"].to_numpy(np.float32))
        weight.append(df["weight"].to_numpy(np.float32))
        sizes.append(len(df))
    s1, score, weight = np.concatenate(s1), np.concatenate(score), np.concatenate(weight)
    codes, inv = np.unique(s1, return_inverse=True)
    n = np.bincount(inv, weights=weight).astype(np.float32)[inv]
    rank = _grouped_rank(inv, score, weight)
    del s1, score, weight, codes, inv
    bounds = np.cumsum([0] + sizes)
    return [pd.DataFrame({"s1_n_cands": n[a:b], "s1_rank": rank[a:b]}) for a, b in zip(bounds[:-1], bounds[1:])]


def featurize(df: pd.DataFrame, s1_norm: NormTable, rec_norm: NormTable, s1_ctx: pd.DataFrame) -> pd.DataFrame:
    """df = one pruned part (whole records). Returns MODEL_FEATURES as float32, row-aligned with df."""
    s1k, rk = id_key(df["s1_id"].to_numpy(object)), id_key(df["cand_id"].to_numpy(object))
    f = pair_features_frame(s1_norm.frame(s1k), rec_norm.frame(rk)).reset_index(drop=True)
    for c in BLOCK_COLS:
        f[c] = df[c].to_numpy(np.float32)
    rec = np.unique(rk, return_inverse=True)[1]
    f["rec_n_cands"] = np.bincount(rec)[rec].astype(np.float32)
    for c in RANK_SCORES:
        v = f[c].to_numpy(np.float32)
        f[f"rec_rank_{c}"] = _grouped_rank(rec, v)
        f[f"rec_gap_{c}"] = (_group_max(rec, v) - v).astype(np.float32)
    f["s1_n_cands"] = s1_ctx["s1_n_cands"].to_numpy()
    f["s1_rank"] = s1_ctx["s1_rank"].to_numpy()
    f["is_s3"] = (rk // 10**12 == 3).astype(np.float32)
    return f[MODEL_FEATURES].astype(np.float32)


def iter_features(kind: str, skip: frozenset[str] = frozenset()):
    """Yields (path, part frame, feature frame) for every part, country by country. Parts named in skip are not
    featurized, but still count in their country's S1 context (it needs every pair pointing at an S1)."""
    split = {"val": "train", "test": "test"}[kind]
    by_country: dict[str, list[Path]] = {}
    for p in part_paths(kind):
        by_country.setdefault(part_country(p), []).append(p)
    for country, cps in by_country.items():
        if all(p.name in skip for p in cps):
            log(f"{kind} {country}: all {len(cps)} parts already done, skipped")
            continue
        ctx = dict(zip(cps, s1_context(kind, cps)))  # per country: an S1 only has candidates of its country
        log(f"{kind} {country}: S1 context for {len(cps)} parts")
        todo = [p for p in cps if p.name not in skip]
        for g in range(0, len(todo), NORM_GROUP):  # norm records for a group of parts at a time (memory)
            group = todo[g:g + NORM_GROUP]
            s1k, rk = [], []
            for p in group:
                t = pq.read_table(p, columns=["s1_id", "cand_id"])
                s1k.append(np.unique(id_key(t["s1_id"])))
                rk.append(np.unique(id_key(t["cand_id"])))
            s1_norm = NormTable(split, ("source1",), np.concatenate(s1k))
            rec_norm = NormTable(split, ("source2", "source3"), np.concatenate(rk))
            log(f"{kind} {country}: norm records loaded for parts {g + 1}-{g + len(group)} of {len(todo)} "
                f"({len(s1_norm.keys):,} S1, {len(rec_norm.keys):,} records)")
            for p in group:
                df = read_part(kind, p)
                yield p, df, featurize(df, s1_norm, rec_norm, ctx.pop(p))
            del s1_norm, rec_norm


def build_val_features() -> None:
    """Resumable: a part whose feature file is newer than its val part and than the code that produced it is kept."""
    out = CACHE / "feats_val"
    out.mkdir(parents=True, exist_ok=True)
    code = max((ROOT / "src" / f).stat().st_mtime for f in ("features.py", "normalize.py", "matcher.py"))
    done = frozenset(p.name for p in part_paths("val")
                     if (out / p.name).exists() and (out / p.name).stat().st_mtime > max(p.stat().st_mtime, code))
    if done:
        log(f"resuming: {len(done)} feature parts already built")
    t0 = time.perf_counter()
    for p, df, f in iter_features("val", skip=done):
        t = pa.Table.from_pandas(pd.concat([df[META_COLS].reset_index(drop=True), f], axis=1), preserve_index=False)
        pq.write_table(t, out / f"{p.stem}.tmp")
        (out / f"{p.stem}.tmp").replace(out / f"{p.stem}.parquet")
        log(f"features {p.name}: {len(df):,} pairs ({time.perf_counter() - t0:.0f} s)")


# ---------- model ----------

def load_side(side: str, columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Feature matrix (float32, preallocated: no concat copy) and meta columns of one split side."""
    paths = sorted((CACHE / "feats_val").glob("*.parquet"))
    n = sum(pc.sum(pc.equal(pq.read_table(p, columns=["side"])["side"], side)).as_py() for p in paths)
    x = np.empty((n, len(MODEL_FEATURES)), np.float32)
    metas, at = [], 0
    for p in paths:
        t = pq.read_table(p, filters=[("side", "=", side)])
        x[at:at + t.num_rows] = np.column_stack([t[c].to_numpy() for c in MODEL_FEATURES])
        metas.append(t.select(columns).to_pandas())
        at += t.num_rows
    return x, pd.concat(metas, ignore_index=True)


def train() -> None:
    import lightgbm as lgb
    t0 = time.perf_counter()
    x, meta = load_side("train", ["cand_id", "label"])
    y = meta["label"].to_numpy(np.int8)
    rec = id_key(meta["cand_id"].to_numpy(object))
    del meta
    urec = np.unique(rec)
    hold = isin_sorted(rec, np.sort(np.random.default_rng(42).choice(urec, len(urec) // 10, replace=False)))
    keep = ~hold & ((y == 1) | (np.random.default_rng(42).random(len(y)) < NEG_RATE))
    log(f"train rows {keep.sum():,} (holdout {hold.sum():,}, negative rate {NEG_RATE}), positives {y[keep].mean():.3f}")
    dtr = lgb.Dataset(x[keep], y[keep], feature_name=MODEL_FEATURES, free_raw_data=True).construct()
    dva = lgb.Dataset(x[hold], y[hold], reference=dtr, free_raw_data=True).construct()
    del x
    log("datasets built")
    evals: dict = {}
    booster = lgb.train(PARAMS, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100),
                                   lgb.record_evaluation(evals)])
    MODELS.mkdir(exist_ok=True)
    booster.save_model(str(MODEL_PATH), num_iteration=booster.best_iteration)
    summary = {"best_iteration": booster.best_iteration, "holdout_logloss": evals["holdout"]["binary_logloss"][booster.best_iteration - 1],
               "train_rows": int(keep.sum()), "holdout_rows": int(hold.sum()), "neg_rate": NEG_RATE,
               "runtime_s": round(time.perf_counter() - t0, 1), "peak_mb": round(peak_rss_mb()),
               "model_mb": round(MODEL_PATH.stat().st_size / 2**20, 1)}
    (MODELS / "train_summary.json").write_text(json.dumps(summary, indent=1))
    write_importance(booster, summary)
    log(json.dumps(summary))


def write_importance(booster, summary: dict) -> None:
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain"), "splits": booster.feature_importance("split")})
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False)
    lines = ["# LightGBM feature importance (generated by `python -m src.matcher train`)", "",
             f"Model: `{MODEL_PATH.relative_to(ROOT).as_posix()}`, {summary['best_iteration']} trees, "
             f"{PARAMS['num_leaves']} leaves, trained on {summary['train_rows']:,} train-side pairs of the val "
             f"split (negative rate {summary['neg_rate']}), early stopping on {summary['holdout_rows']:,} "
             f"holdout pairs (10% of train-side records). Features computed with the current "
             f"normalize.py / features.py.", "",
             "| rank | feature | gain share | splits |", "|---|---|---|---|"]
    lines += [f"| {i} | {r.feature} | {r.gain_share:.4f} | {r.splits:,} |" for i, r in enumerate(imp.itertuples(), 1)]
    (ROOT / "docs" / "feature_importance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODEL_PATH))


def predict(booster, f: pd.DataFrame) -> np.ndarray:
    return booster.predict(f[MODEL_FEATURES].to_numpy(np.float32)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "train"])
    ap.add_argument("kind", nargs="?", default="val", choices=["val"])
    args = ap.parse_args()
    start_memory_guard()
    build_val_features() if args.cmd == "features" else train()
    log("done")


if __name__ == "__main__":
    main()
