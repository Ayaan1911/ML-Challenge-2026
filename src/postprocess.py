"""Decision layer: one S1 per record, probability threshold, per-S1 matched lists; threshold sweep on the eval side.

Decision (EDA: every S2/S3 record belongs to at most one S1): keep each record only for its highest-probability
S1, and only if that probability >= t.

Eval side of the val split (src/valsplit.py): records of half-B val S1s, all queried, plus half of the 20%
sample of other records. False positives from those other records are therefore under-counted about 10x
(1 / (0.2 * 0.5)). Corrected F0.5 counts each of them as 10 false positives: per S1,
F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP_val + 10 FP_other). t is chosen on the corrected score.

Pass 8 adds optional rules on top of that (class Decider), tuned jointly on the corrected F0.5 by `tune`.

Run: python -m src.postprocess sweep    -> models/threshold.json, docs/matcher_report.md
     python -m src.postprocess tune     -> models/decision.json, docs/decision_report.md (read by src.run_pipeline)
     python -m src.postprocess errors   -> docs/errors_val.csv
"""
import argparse
import itertools
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
DECISION_PATH = ROOT / "models" / "decision.json"
OFF = {"t2": None, "t3": None, "m": None, "k": None}
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


class Decider:
    """Decision rules over one set of scored pairs; the grouping is computed once, then any rule set is applied.

    base  t       keep a record only for its highest-probability S1, and only if prob >= t (= decide())
    b     t2      second chance: an S1 left with no match takes its single best candidate if prob >= t2 and that
                  record is matched to no S1 (a record wanted by several such S1s goes to the highest prob)
    c     t3, m   relative: also keep a best-S1 row with prob >= t3 and prob >= (top accepted prob of its S1) - m
    d     k       cap: at most k matches per S1, highest prob first (applied before b)
    A rule whose parameters are None is off. The per-S1 grouping is built only when b, c or d is on, so the
    threshold alone costs no more memory than decide().
    """

    def __init__(self, s1: np.ndarray, rec: np.ndarray, prob: np.ndarray):
        self.s1, self.rec, self.prob = s1, rec, prob
        self.best = best_per_record(rec, prob)
        self.g = None

    def _group(self) -> None:
        prob = self.prob
        self.g = np.unique(self.s1, return_inverse=True)[1].astype(np.int32)
        self.ri = np.unique(self.rec, return_inverse=True)[1].astype(np.int32)
        self.n_g, self.n_r = int(self.g.max(initial=-1)) + 1, int(self.ri.max(initial=-1)) + 1
        order = np.lexsort((-prob, self.g))  # by S1, then prob descending
        go = self.g[order]
        first = np.r_[True, go[1:] != go[:-1]]
        self.top = np.zeros(self.n_g, np.int64)  # every S1 here has at least one row
        self.top[go[first]] = order[first]
        ob = order[self.best[order]]
        del order, go
        gb = self.g[ob]
        fb = np.r_[True, gb[1:] != gb[:-1]]
        self.top_best = np.full(self.n_g, -np.inf, np.float32)  # highest best-S1 prob per S1
        self.top_best[gb[fb]] = prob[ob[fb]]
        pos = np.arange(len(ob))
        self.rank = np.full(len(prob), np.iinfo(np.int32).max, np.int32)  # rank among the S1's best-S1 rows
        self.rank[ob] = pos - np.maximum.accumulate(np.where(fb, pos, 0))

    def __call__(self, t: float, t2=None, t3=None, m=None, k=None) -> np.ndarray:
        sel = self.best & (self.prob >= t)
        if t2 is None and t3 is None and k is None:
            return sel
        if self.g is None:
            self._group()
        if t3 is not None:
            floor = np.where(self.top_best >= t, np.maximum(t3, self.top_best - m), np.inf).astype(np.float32)
            sel |= self.best & (self.prob >= floor[self.g])
        if k is not None:
            sel &= self.rank < k
        if t2 is not None:
            c = self.top[np.bincount(self.g[sel], minlength=self.n_g) == 0]
            c = c[self.prob[c] >= t2]
            taken = np.zeros(self.n_r, bool)
            taken[self.ri[sel]] = True
            c = c[~taken[self.ri[c]]]
            sel[c[best_per_record(self.ri[c], self.prob[c])]] = True
        return sel


def load_rules() -> dict:
    """models/decision.json (python -m src.postprocess tune) if present, else the plain threshold."""
    if DECISION_PATH.exists():
        return {**OFF, **json.loads(DECISION_PATH.read_text())["rules"]}
    return {**OFF, "t": json.loads(THRESHOLD_PATH.read_text())["t"]}


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
        self.decider = Decider(self.s1, self.rec, self.prob)
        self.best = self.decider.best

    def counts(self, rules: float | dict):
        rules = {**OFF, **rules} if isinstance(rules, dict) else {**OFF, "t": rules}
        sel = self.decider(**rules) & self.in_b
        n = len(self.b_keys)
        tp = np.bincount(self.idx[sel & self.label], minlength=n).astype(np.float64)
        fp_raw = np.bincount(self.idx[sel & ~self.label], minlength=n).astype(np.float64)
        fp_w = np.bincount(self.idx[sel & ~self.label], weights=self.fpw[sel & ~self.label], minlength=n)
        return sel, tp, fp_raw, fp_w

    def score(self, rules: float | dict) -> dict:
        _, tp, fp_raw, fp_w = self.counts(rules)
        fn = self.n_true - tp
        raw, cor = f05_per_s1(tp, fn, fp_raw), f05_per_s1(tp, fn, fp_w)
        single = self.n_true == 0
        rules = rules if isinstance(rules, dict) else {"t": float(rules)}
        out = {**rules, "f05": raw.mean(), "f05_corrected": cor.mean(),
               "precision": tp.sum() / max(tp.sum() + fp_raw.sum(), 1),
               "precision_corrected": tp.sum() / max(tp.sum() + fp_w.sum(), 1),
               "recall": tp.sum() / self.n_true.sum(),
               "singleton_accuracy": float((fp_raw[single] == 0).mean()) if single.any() else float("nan"),
               "n_s1": len(self.b_keys), "n_singletons": int(single.sum())}
        for c in sorted(set(self.country)):
            m = self.country == c
            out[f"f05[{c}]"], out[f"f05_corrected[{c}]"], out[f"n_s1[{c}]"] = raw[m].mean(), cor[m].mean(), int(m.sum())
        return out

    def matches(self, rules: float | dict) -> dict[str, set[str]]:
        sel = self.counts(rules)[0]
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


# ---------- decision rules (Pass 8) ----------

RULE_PARAMS = {"b": ("t2",), "c": ("t3", "m"), "d": ("k",)}  # t (rule a) is tuned with every subset
COARSE = {"t": np.round(np.arange(0.70, 0.91, 0.02), 2), "t2": np.round(np.arange(0.20, 0.80, 0.05), 2),
          "t3": np.round(np.arange(0.20, 0.80, 0.05), 2), "m": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
          "k": [1, 2, 3, 4, 5, 6, 8, 12]}
FINE = {"t": (0.02, 0.005), "t2": (0.05, 0.01), "t3": (0.05, 0.01), "m": (0.03, 0.01), "k": (2, 1)}  # (+-span, step)
START = {"t": 0.80, "t2": 0.60, "t3": 0.60, "m": 0.10, "k": 4}
SUBSETS = ["", "b", "c", "d", "bc", "bd", "cd", "bcd"]
MIN_GAIN = 0.001  # a rule is adopted only if dropping it costs at least this much corrected F0.5


class Tuner:
    """Cached corrected-F0.5 lookups; parameter sets that break t2 < t or t3 < t score -inf."""

    def __init__(self, ev: Eval):
        self.ev, self.cache = ev, {}

    def score(self, r: dict) -> dict:
        key = tuple(r[k] for k in ("t", *OFF))
        if key not in self.cache:
            self.cache[key] = self.ev.score(r)
        return self.cache[key]

    def f(self, r: dict) -> float:
        if (r["t2"] is not None and r["t2"] >= r["t"]) or (r["t3"] is not None and r["t3"] >= r["t"]):
            return -np.inf
        return self.score(r)["f05_corrected"]

    def best(self, r: dict, names: tuple, grids: list) -> dict:
        cands = [{**r, **{n: (int(v) if n == "k" else round(float(v), 3)) for n, v in zip(names, vals)}}
                 for vals in itertools.product(*grids)]
        return max([r] + cands, key=self.f)  # current values win ties

    def tune(self, rules: str) -> dict:
        """Coordinate descent: coarse grids ((t3, m) jointly), then fine steps around the optimum."""
        params = ["t"] + [p for x in rules for p in RULE_PARAMS[x]]
        r = {**OFF, **{p: START[p] for p in params}}
        for grid in ("coarse", "fine"):
            for _ in range(5):
                before = dict(r)
                if grid == "coarse":
                    for block in [("t",)] + [RULE_PARAMS[x] for x in rules]:
                        r = self.best(r, block, [COARSE[p] for p in block])
                else:
                    for p in params:
                        span, step = FINE[p]
                        lo = max(r[p] - span, step)
                        r = self.best(r, (p,), [np.arange(lo, r[p] + span + step / 2, step)])
                if r == before:
                    break
        return r


def describe(r: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in r.items() if k in ("t", *OFF) and v is not None)


def tune() -> None:
    t0 = time.perf_counter()
    ev = load_eval()
    tn = Tuner(ev)
    p7 = {**OFF, "t": json.loads(THRESHOLD_PATH.read_text())["t"]}
    res = {}
    for s in SUBSETS:
        res[s] = tn.tune(s)
        log(f"rules a{('+' + '+'.join(s)) if s else ''}: {describe(res[s])} -> corrected F0.5 "
            f"{tn.f(res[s]):.5f} ({len(tn.cache)} evaluations so far)")
    f = {s: tn.f(res[s]) for s in SUBSETS}
    chosen = max(SUBSETS, key=lambda s: (f[s], -len(s)))
    while chosen:  # drop rules that do not earn MIN_GAIN, cheapest first
        drop, loss = min(((chosen.replace(x, ""), f[chosen] - f[chosen.replace(x, "")]) for x in chosen),
                         key=lambda d: d[1])
        if loss >= MIN_GAIN:
            break
        chosen = drop
    final = res[chosen] if chosen or f[""] - tn.f(p7) >= MIN_GAIN else p7
    best = tn.score(final)
    ref = macro_f05(ev.matches(final), load_truth(), ev.b_ids)
    if abs(ref - best["f05"]) > 1e-9:
        raise SystemExit(f"vectorized F0.5 {best['f05']} != src.metric {ref}")
    rules = {k: v for k, v in final.items() if k in ("t", *OFF) and v is not None}
    DECISION_PATH.write_text(json.dumps({"rules": rules, "adopted": "a" + chosen, "chosen_on": "f05_corrected",
                                         "min_gain": MIN_GAIN, "eval": {k: v for k, v in best.items() if k not in rules
                                                                        and k not in OFF}}, indent=1))
    rows = [("Pass 7 as is", p7)] + [(("a+" + "+".join(s)) if s else "a (t re-swept)", res[s]) for s in SUBSETS]
    write_decision_report(tn, rows, ("chosen: a" + ("+" + "+".join(chosen) if chosen else ""), final),
                          round(time.perf_counter() - t0, 1))
    log(f"chosen {describe(final)}: " + json.dumps(best))


def write_decision_report(tn: Tuner, rows: list, chosen: tuple, runtime_s: float) -> None:
    base = tn.f(rows[0][1])
    countries = sorted(k[4:-1] for k in tn.score(rows[0][1]) if k.startswith("f05["))
    L = ["# Decision rules (generated by `python -m src.postprocess tune`)", "",
         "Tuned on the saved eval-side predictions of the current model (same eval side and corrected F0.5 as "
         "docs/matcher_report.md). Rules (src/postprocess.py, class Decider): **a** threshold t on the record's "
         "best S1; **b** second chance, an S1 with no match takes its best candidate if prob >= t2 and that "
         "record is matched nowhere; **c** relative, also keep best-S1 rows with prob >= t3 within m of the "
         "S1's top accepted prob; **d** at most k matches per S1. Each subset is tuned jointly with t by "
         f"coordinate descent (coarse grid, then fine). A rule is adopted only if dropping it costs at least "
         f"{MIN_GAIN} corrected F0.5; simpler subsets win ties.", "",
         "| rules | parameters | F0.5 corrected | F0.5 | " + " | ".join(f"{c} corrected | {c}" for c in countries)
         + " | precision corrected | recall | singleton acc | vs Pass 7 |",
         "|---|---|---|---|" + "---|" * (2 * len(countries)) + "---|---|---|---|"]
    for name, r in rows + [chosen]:
        x = tn.score(r)
        bold = "**" if (name, r) == chosen else ""
        L.append(f"| {bold}{name}{bold} | {describe(r)} | {bold}{x['f05_corrected']:.4f}{bold} | {x['f05']:.4f} | "
                 + " | ".join(f"{x[f'f05_corrected[{c}]']:.4f} | {x[f'f05[{c}]']:.4f}" for c in countries)
                 + f" | {x['precision_corrected']:.4f} | {x['recall']:.4f} | {x['singleton_accuracy']:.4f} | "
                 f"{x['f05_corrected'] - base:+.4f} |")
    L += ["", f"{len(tn.cache):,} rule sets evaluated in {runtime_s:.0f} s, peak {peak_rss_mb():,.0f} MB."]
    (ROOT / "docs" / "decision_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


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
    rules = load_rules()
    t = rules["t"]
    sel = ev.counts(rules)[0]
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
    err["rules"] = json.dumps({k: v for k, v in rules.items() if v is not None})
    err.to_csv(ROOT / "docs" / "errors_val.csv", index=False, encoding="utf-8")
    log(f"errors_val.csv: {err['error_type'].value_counts().to_dict()} (all FN types: "
        f"{fn['error_type'].value_counts().to_dict()})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "tune", "errors"])
    args = ap.parse_args()
    start_memory_guard()
    {"sweep": sweep, "tune": tune, "errors": errors}[args.cmd]()


if __name__ == "__main__":
    main()
