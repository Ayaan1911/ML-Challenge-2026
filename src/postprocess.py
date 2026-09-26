"""Decision layer: one S1 per record, probability threshold, per-S1 matched lists; threshold sweep on the eval side.

Decision (EDA: every S2/S3 record belongs to at most one S1): keep each record only for its highest-probability
S1, and only if that probability >= t.

Eval side of the val split (src/valsplit.py): records of half-B val S1s, all queried, plus half of the 20%
sample of other records. False positives from those other records are therefore under-counted about 10x
(1 / (0.2 * 0.5)). Corrected F0.5 counts each of them as 10 false positives: per S1,
F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP_val + 10 FP_other). t is chosen on the corrected score.

Run: python -m src.postprocess sweep    -> models/threshold.json, docs/matcher_report.md
     python -m src.postprocess errors   -> docs/errors_val.csv
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CACHE, ROOT, id_key, key_id, load_truth, peak_rss_mb, start_memory_guard
from src.metric import macro_f05
from src.norm_cache import norm_path
from src.split import SPLITS

THRESHOLD_PATH = ROOT / "models" / "threshold.json"
PRED_EVAL = CACHE / "pred_eval.parquet"
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)
OTHER_FP_WEIGHT = 10.0  # = 1 / (0.2 val sampling * 0.5 eval half); equals 2 * the stored weight of 5


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


def best_per_record(rec: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """True on the single highest-probability row of each record (ties: first row)."""
    order = np.lexsort((-prob, rec))
    first = np.r_[True, rec[order][1:] != rec[order][:-1]]
    mask = np.zeros(len(rec), bool)
    mask[order[first]] = True
    return mask


def decide(rec: np.ndarray, prob: np.ndarray, t: float, best: np.ndarray | None = None) -> np.ndarray:
    return (best_per_record(rec, prob) if best is None else best) & (prob >= t)


def f05_per_s1(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Competition F0.5 per S1 from counts; true-empty & predicted-empty = 1.0 (den == 0)."""
    den = 1.25 * tp + 0.25 * fn + fp
    return np.where(den == 0, 1.0, 1.25 * tp / np.where(den == 0, 1, den))


class Eval:
    """Eval-side predictions scored over the half-B S1 ids."""

    def __init__(self, pred: pd.DataFrame, b_ids: list[str], truth: dict[str, set[str]], country: pd.Series):
        keys = id_key(list(b_ids))
        order = np.argsort(keys)  # key order, not string order ("S1-10" < "S1-9" as strings)
        self.b_ids, self.b_keys = [b_ids[i] for i in order], keys[order]
        self.n_true = np.array([len(truth[s]) for s in self.b_ids], np.float64)
        self.country = country.reindex(self.b_ids).fillna("").to_numpy(object)
        self.rec = id_key(pred["cand_id"].to_numpy(object))
        self.s1 = id_key(pred["s1_id"].to_numpy(object))
        self.prob = pred["prob"].to_numpy(np.float32)
        self.label = pred["label"].to_numpy().astype(bool)
        self.fpw = np.where(pred["query_group"].to_numpy(object) == "other_sampled", OTHER_FP_WEIGHT, 1.0)
        i = np.searchsorted(self.b_keys, self.s1).clip(max=len(self.b_keys) - 1)
        self.in_b = self.b_keys[i] == self.s1
        self.idx = i
        self.best = best_per_record(self.rec, self.prob)

    def counts(self, t: float):
        sel = decide(self.rec, self.prob, t, self.best) & self.in_b
        n = len(self.b_keys)
        tp = np.bincount(self.idx[sel & self.label], minlength=n).astype(np.float64)
        fp_raw = np.bincount(self.idx[sel & ~self.label], minlength=n).astype(np.float64)
        fp_w = np.bincount(self.idx[sel & ~self.label], weights=self.fpw[sel & ~self.label], minlength=n)
        return sel, tp, fp_raw, fp_w

    def score(self, t: float) -> dict:
        _, tp, fp_raw, fp_w = self.counts(t)
        fn = self.n_true - tp
        raw, cor = f05_per_s1(tp, fn, fp_raw), f05_per_s1(tp, fn, fp_w)
        single = self.n_true == 0
        out = {"t": float(t), "f05": raw.mean(), "f05_corrected": cor.mean(),
               "precision": tp.sum() / max(tp.sum() + fp_raw.sum(), 1),
               "precision_corrected": tp.sum() / max(tp.sum() + fp_w.sum(), 1),
               "recall": tp.sum() / self.n_true.sum(),
               "singleton_accuracy": float((fp_raw[single] == 0).mean()) if single.any() else float("nan"),
               "n_s1": len(self.b_keys), "n_singletons": int(single.sum())}
        for c in sorted(set(self.country)):
            m = self.country == c
            out[f"f05[{c}]"], out[f"f05_corrected[{c}]"], out[f"n_s1[{c}]"] = raw[m].mean(), cor[m].mean(), int(m.sum())
        return out

    def matches(self, t: float) -> dict[str, set[str]]:
        sel = self.counts(t)[0]
        ids = key_id(self.rec[sel]).to_pylist()
        out: dict[str, set[str]] = {}
        for s, r in zip(np.array(self.b_ids, object)[self.idx[sel]], ids):
            out.setdefault(s, set()).add(r)
        return out


# ---------- eval predictions ----------

def predict_eval() -> pd.DataFrame:
    from src.matcher import load_model, predict
    booster = load_model()
    frames = []
    for p in sorted((CACHE / "feats_val").glob("*.parquet")):
        t = pq.read_table(p, filters=[("side", "=", "eval")]).to_pandas()
        t["prob"] = predict(booster, t)
        frames.append(t[["s1_id", "cand_id", "country", "label", "query_group", "prob"]])
    pred = pd.concat(frames, ignore_index=True)
    pred.to_parquet(PRED_EVAL.with_suffix(".tmp"), index=False)
    PRED_EVAL.with_suffix(".tmp").replace(PRED_EVAL)
    return pred


def load_eval() -> Eval:
    from src.matcher import MODEL_PATH
    fresh = PRED_EVAL.exists() and PRED_EVAL.stat().st_mtime > MODEL_PATH.stat().st_mtime
    pred = pd.read_parquet(PRED_EVAL) if fresh else predict_eval()
    log(f"eval predictions: {len(pred):,} pairs")
    b_ids = (SPLITS / "valsplit_B_s1_ids.txt").read_text(encoding="utf-8").split()
    truth = load_truth()
    s1 = pq.read_table(norm_path("train", "source1"), columns=["entity_id", "country_norm"],
                       filters=[("entity_id", "in", b_ids)]).to_pandas()
    return Eval(pred, b_ids, {s: truth[s] for s in b_ids}, s1.set_index("entity_id")["country_norm"])


def sweep() -> None:
    t0 = time.perf_counter()
    ev = load_eval()
    rows = [ev.score(t) for t in GRID]
    best = max(rows, key=lambda r: r["f05_corrected"])
    best_raw = max(rows, key=lambda r: r["f05"])
    # check the vectorized scorer against the reference metric at the chosen threshold
    truth = load_truth()
    ref = macro_f05(ev.matches(best["t"]), truth, ev.b_ids)
    if abs(ref - best["f05"]) > 1e-9:
        raise SystemExit(f"vectorized F0.5 {best['f05']} != src.metric {ref}")
    THRESHOLD_PATH.parent.mkdir(exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps({"t": best["t"], "chosen_on": "f05_corrected", **best}, indent=1))
    write_report(rows, best, best_raw, round(time.perf_counter() - t0, 1))
    log(json.dumps(best))


def write_report(rows: list[dict], best: dict, best_raw: dict, runtime_s: float) -> None:
    countries = sorted(k[4:-1] for k in best if k.startswith("f05[") )
    L = ["# Matcher evaluation (generated by `python -m src.postprocess sweep`)", "",
         "Eval side of the val split: records of the half-B val S1s plus half of the sampled other records, "
         f"scored over {best['n_s1']:,} half-B S1 ids ({best['n_singletons']:,} singletons). F0.5 is the "
         "competition macro F0.5 over those S1s, including true pairs that blocking or pruning lost. "
         f"**Corrected** counts each false positive from an 'other' record as {OTHER_FP_WEIGHT:.0f} "
         "(those records were sampled at 20% and then halved). It cannot add back the singletons that the "
         "unqueried 90% would have hit, and records of half-A S1s never appear on the eval side, so even the "
         "corrected score is somewhat optimistic.", "",
         f"**Chosen threshold t = {best['t']:.2f}** (max corrected F0.5). Uncorrected F0.5 peaks at "
         f"t = {best_raw['t']:.2f} ({best_raw['f05']:.4f}).", "",
         "| metric | uncorrected | corrected |", "|---|---|---|",
         f"| macro F0.5 | {best['f05']:.4f} | **{best['f05_corrected']:.4f}** |"]
    L += [f"| macro F0.5, {c} ({best[f'n_s1[{c}]']:,} S1) | {best[f'f05[{c}]']:.4f} | {best[f'f05_corrected[{c}]']:.4f} |"
          for c in countries]
    L += [f"| pair precision | {best['precision']:.4f} | {best['precision_corrected']:.4f} |",
          f"| pair recall | {best['recall']:.4f} | {best['recall']:.4f} |",
          f"| singleton accuracy (no prediction) | {best['singleton_accuracy']:.4f} | - |", "",
          "## Threshold sweep", "", "| t | F0.5 | F0.5 corrected | precision | precision corrected | recall | singleton acc |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['t']:.2f} | {r['f05']:.4f} | {r['f05_corrected']:.4f} | {r['precision']:.4f} | "
          f"{r['precision_corrected']:.4f} | {r['recall']:.4f} | {r['singleton_accuracy']:.4f} |"
          for r in rows if round(r["t"] * 100) % 5 == 0 or r["t"] in (best["t"], best_raw["t"])]
    L += ["", f"Sweep runtime {runtime_s:.0f} s, peak {peak_rss_mb():,.0f} MB."]
    (ROOT / "docs" / "matcher_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------- error sample ----------

def _rows(split: str, ids: list[str]) -> pd.DataFrame:
    """Raw and normalized fields of the given ids from all three sources."""
    raw, norm = [], []
    for s in ("source1", "source2", "source3"):
        raw.append(pq.read_table(CACHE / f"{split}_{s}.parquet", filters=[("entity_id", "in", ids)]).to_pandas())
        norm.append(pq.read_table(norm_path(split, s), columns=["entity_id", "name_norm", "name_core", "addr_norm", "country_norm"],
                                  filters=[("entity_id", "in", ids)]).to_pandas())
    return pd.concat(raw).merge(pd.concat(norm), on="entity_id", how="left").set_index("entity_id")


def errors(n: int = 3000) -> None:
    ev = load_eval()
    t = json.loads(THRESHOLD_PATH.read_text())["t"]
    sel = ev.counts(t)[0]
    rng = np.random.default_rng(42)
    fp = np.flatnonzero(sel & ~ev.label)
    fp = fp[rng.choice(len(fp), min(n, len(fp)), replace=False)]
    b_ids = np.array(ev.b_ids, object)
    fps = pd.DataFrame({"error_type": "false_positive", "s1_id": b_ids[ev.idx[fp]],
                        "record_id": key_id(ev.rec[fp]).to_pylist(), "prob": ev.prob[fp],
                        "record_group": np.where(ev.fpw[fp] > 1, "other_sampled", "val_record")})
    # false negatives: every true (B S1, record) pair that was not predicted
    truth = load_truth()
    true_pairs = pd.DataFrame([(s, r) for s in ev.b_ids for r in truth[s]], columns=["s1_id", "record_id"])
    pairs = pd.DataFrame({"s1_id": b_ids[ev.idx[ev.in_b]], "record_id": key_id(ev.rec[ev.in_b]).to_pylist(),
                          "prob": ev.prob[ev.in_b], "best": ev.best[ev.in_b], "sel": sel[ev.in_b]})
    fn = true_pairs.merge(pairs, on=["s1_id", "record_id"], how="left")
    fn = fn[fn["sel"] != True]  # noqa: E712 (NaN = not a candidate)
    fn["error_type"] = np.select([fn["prob"].isna(), fn["best"] == True], ["fn_not_candidate", "fn_below_threshold"],  # noqa: E712
                                 "fn_lost_to_other_s1")
    fn = fn.sample(min(n, len(fn)), random_state=42)[["error_type", "s1_id", "record_id", "prob"]]
    err = pd.concat([fps, fn], ignore_index=True)
    info = _rows("train", sorted(set(err["s1_id"]) | set(err["record_id"])))
    cols = ["business_name", "business_address", "country", "name_norm", "name_core", "addr_norm", "country_norm"]
    for side, idcol in (("s1", "s1_id"), ("rec", "record_id")):
        err = err.join(info[cols].add_prefix(f"{side}_"), on=idcol)
    err["threshold"] = t
    err.to_csv(ROOT / "docs" / "errors_val.csv", index=False, encoding="utf-8")
    log(f"errors_val.csv: {err['error_type'].value_counts().to_dict()} (all FN types: "
        f"{fn['error_type'].value_counts().to_dict()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "errors"])
    args = ap.parse_args()
    start_memory_guard()
    sweep() if args.cmd == "sweep" else errors()


if __name__ == "__main__":
    main()
