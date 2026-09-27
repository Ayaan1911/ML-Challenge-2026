"""Pass 10: where the eval F0.5 is lost, structure checks on train, and a sibling-propagation rescue.
Read-only on every saved output: uses data/cache/pred_eval.parquet + pred_eval_chainfix_delta.parquet (chainfix eval
decision) and the norm caches.

Run: python -m src.pass10 loss       -> loss breakdown by cause (eval side, chainfix, t = 0.80)
     python -m src.pass10 structure  -> (a) row position, (b) id structure: train, vs shuffled pairs
     python -m src.pass10 siblings   -> (c) sibling signal, (d) within-source duplicates, rescue tuning on eval
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src.data import CACHE, id_key, key_id, load_truth, load_truth_df, peak_rss_mb, start_memory_guard
from src.norm_cache import norm_path
from src.postprocess import OTHER_FP_WEIGHT, PRED_EVAL, Eval, f05_per_s1
from src.split import SPLITS

T = 0.80
rng = np.random.default_rng(42)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


def load_chainfix_eval() -> Eval:
    """Eval-side predictions with the chainfix delta swapped in (every row of the affected records replaced)."""
    pe = pd.read_parquet(PRED_EVAL, columns=["s1_id", "cand_id", "label", "query_group", "prob"])
    delta = pd.read_parquet(CACHE / "pred_eval_chainfix_delta.parquet", columns=pe.columns.tolist())
    pe = pd.concat([pe[~pe["cand_id"].isin(set(delta["cand_id"]))], delta], ignore_index=True)
    b_ids = (SPLITS / "valsplit_B_s1_ids.txt").read_text(encoding="utf-8").split()
    truth = load_truth()
    s1c = pq.read_table(norm_path("train", "source1"), columns=["entity_id", "country_norm"],
                        filters=[("entity_id", "in", b_ids)]).to_pandas().set_index("entity_id")["country_norm"]
    ev = Eval(pe, b_ids, {s: truth[s] for s in b_ids}, s1c)
    ev.truth = {s: truth[s] for s in b_ids}
    return ev


# ---------- 1. loss breakdown ----------

def loss_breakdown(ev: Eval) -> dict:
    sel, tp, fp_raw, fp_w = ev.counts(T)
    n = len(ev.b_keys)
    fn = ev.n_true - tp
    true_rows = ev.in_b & ev.label
    has_true_cand = np.bincount(ev.idx[true_rows], minlength=n) > 0
    true_best = np.bincount(ev.idx[true_rows & ev.best], minlength=n) > 0  # a true record's best S1 is this one
    single = ev.n_true == 0
    out = {}
    for name, fp in (("raw", fp_raw), ("corrected", fp_w)):
        f = f05_per_s1(tp, fn, fp)
        f_nofp = f05_per_s1(tp, fn, np.zeros(n))
        zero = ~single & (tp == 0)
        buckets = {
            "singleton scored 0 (false positive on an S1 with no match)": single & (fp > 0),
            "S1 scored 0: no true record was a candidate": zero & ~has_true_cand,
            "S1 scored 0: a true candidate's best S1 is this one but p < t": zero & has_true_cand & true_best,
            "S1 scored 0: all true candidates went to (or scored best at) another S1": zero & has_true_cand & ~true_best,
        }
        rows = {k: {"s1": int(m.sum()), "points": float((1 - f)[m].sum() / n)} for k, m in buckets.items()}
        part = ~single & (tp > 0)
        rows["partial recall (TP > 0, missed true records)"] = {"s1": int((part & (fn > 0)).sum()),
                                                               "points": float((1 - f_nofp)[part].sum() / n)}
        rows["false positives on matched S1 (TP > 0)"] = {"s1": int((part & (fp > 0)).sum()),
                                                          "points": float((f_nofp - f)[part].sum() / n)}
        rows["total lost"] = {"s1": int((f < 1).sum()), "points": float((1 - f).mean())}
        out[name] = {"f05": float(f.mean()), "buckets": rows}
    # causes of the missed true pairs inside partial-recall S1
    part_s1 = (~single & (tp > 0) & (fn > 0))
    fn_rows = true_rows & ~sel & part_s1[ev.idx]
    missed = fn[part_s1].sum()
    cand_fn = int(fn_rows.sum())
    below = int((fn_rows & ev.best).sum())
    out["partial_recall_fn_pairs"] = {"missed": int(missed), "not_a_candidate": int(missed - cand_fn),
                                      "best_here_but_below_t": below, "went_to_another_s1": cand_fn - below}
    return out


# ---------- 2a/2b. structure on train ----------

def structure() -> dict:
    gt = load_truth_df()
    pos = {}
    for s in ("source1", "source2", "source3"):
        ids = pq.read_table(CACHE / f"train_{s}.parquet", columns=["entity_id"])["entity_id"].to_numpy(zero_copy_only=False)
        pos[s] = pd.Series(np.arange(len(ids)) / len(ids), index=ids)
    pairs = gt.assign(r=gt["matched_entity_ids"].str.split(",")).explode("r")
    pairs = pairs[pairs["r"].fillna("") != ""][["source1_entity_id", "r"]].rename(columns={"source1_entity_id": "s1"})
    pairs = pairs.sample(min(500_000, len(pairs)), random_state=42)
    out = {}
    rec_pos = pd.concat([pos["source2"], pos["source3"]])
    for kind in ("matched", "shuffled"):
        r = pairs["r"].to_numpy(object) if kind == "matched" else rng.permutation(pairs["r"].to_numpy(object))
        a, b = pos["source1"].reindex(pairs["s1"]).to_numpy(), rec_pos.reindex(r).to_numpy()
        d = np.abs(a - b)
        na, nb = id_key(pairs["s1"].tolist()) % 10**12, id_key(list(r)) % 10**12
        out[kind] = {
            "row: mean |rel pos diff|": float(d.mean()), "row: median": float(np.median(d)),
            "row: share < 0.001": float((d < 0.001).mean()), "row: share < 0.01": float((d < 0.01).mean()),
            "row: spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
            "id: spearman(s1 number, record number)": float(pd.Series(na).corr(pd.Series(nb), method="spearman")),
            "id: same last 3 digits": float((na % 1000 == nb % 1000).mean()),
            "id: same last 2 digits": float((na % 100 == nb % 100).mean()),
            "id: |relative number diff| < 1%": float((np.abs(na - nb) / np.maximum(na, 1) < 0.01).mean()),
            "id: record number > s1 number": float((nb > na).mean()),
        }
    out["id ranges"] = {s: [int(pos[s].index.map(lambda x: int(x.split("-")[1])).min()),
                            int(pos[s].index.map(lambda x: int(x.split("-")[1])).max())] for s in pos}
    return out


# ---------- 2c/2d + rescue ----------

def norm_keys(ids: np.ndarray) -> pd.DataFrame:
    """name_core, addr_core, first house number for record keys (train split)."""
    want = key_id(ids).to_pylist()
    frames = []
    for s in ("source2", "source3"):
        t = pq.read_table(norm_path("train", s), columns=["entity_id", "name_core", "addr_core", "numbers"],
                          filters=[("entity_id", "in", want)]).to_pandas()
        t["num"] = [n[0] if len(n) else "" for n in t.pop("numbers")]
        t["source"] = s
        frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    d["rec"] = id_key(d["entity_id"].tolist())
    return d.set_index("rec")


def record_table(ev: Eval) -> pd.DataFrame:
    """One row per eval record: its chosen S1 (best row), best p, whether selected, true B S1 (if any), weight."""
    order = np.flatnonzero(ev.best)
    rec = pd.DataFrame({"rec": ev.rec[order], "s1": ev.s1[order], "p": ev.prob[order], "w": ev.fpw[order]})
    rec["selected"] = rec["p"] >= T
    b = pd.Series(np.array(ev.b_ids, object))
    true_s1 = {}
    for s, recs in ev.truth.items():
        for r in recs:
            true_s1[r] = s
    ids = key_id(rec["rec"].to_numpy()).to_pylist()
    rec["true_s1"] = [true_s1.get(r, "") for r in ids]
    rec["true_key"] = np.where(rec["true_s1"] != "", id_key([x if x else "S1-0" for x in rec["true_s1"]]), 0)
    return rec.set_index("rec")


def siblings(ev: Eval) -> dict:
    rt = record_table(ev)
    nk = norm_keys(rt.index.to_numpy())
    rt = rt.join(nk[["name_core", "addr_core", "num", "source"]])
    rt["akey"] = np.where((rt["addr_core"] != "") & (rt["num"] != ""), rt["addr_core"] + "|" + rt["num"], "")
    ok = rt["selected"] & (rt["s1"] == rt["true_key"])  # correctly assigned
    fn = (rt["true_s1"] != "") & ~ok  # true record of a B S1 that is not correctly assigned
    out = {"records": len(rt), "fn_records": int(fn.sum())}
    # (c) does an identical-key sibling sit correctly on the true S1?
    for key in ("akey", "name_core"):
        good = rt[ok & (rt[key] != "")].groupby([key, "true_key"]).size().rename("n").reset_index()
        f = rt[fn & (rt[key] != "")].reset_index()[["rec", key, "true_key"]].merge(good, on=[key, "true_key"], how="left")
        hit = f["n"].notna().groupby(f["rec"]).any()
        # random baseline: same FN records, the sibling's S1 compared with a random B S1
        fake = f.drop(columns="n").assign(true_key=rng.choice(ev.b_keys, len(f)))
        fake = fake.merge(good, on=[key, "true_key"], how="left")
        out[f"(c) FN records with a correctly assigned {key} sibling on the true S1"] = {
            "share_of_fn": float(hit.sum() / max(fn.sum(), 1)), "random_s1_baseline": float(fake["n"].notna().mean())}
    # (d) within-source duplicates of one S1
    tr = rt[rt["true_s1"] != ""].reset_index()
    g = tr.groupby(["true_key", "source"])
    multi = tr[g["rec"].transform("size") >= 2]
    dup = multi.groupby(["true_key", "source", "name_core", "addr_core"])["rec"].transform("size") >= 2
    m = multi[dup]
    right = (m["selected"] & (m["s1"] == m["true_key"])).groupby(
        [m["true_key"], m["source"], m["name_core"], m["addr_core"]]).agg(["all", "any"])
    cons = pd.Series(np.select([right["all"], ~right["any"]], ["all assigned", "none assigned"], "split"))
    out["(d) true records sharing S1 and source"] = int(len(multi))
    out["(d) ... of which identical name_core and addr_core to another"] = float(dup.mean()) if len(multi) else 0.0
    out["(d) identical groups: assignment"] = cons.value_counts(normalize=True).round(4).to_dict()
    rt.to_parquet(CACHE / "pass10_records.parquet")  # for the rescue tuner
    return out


def rescue(ev: Eval, rt: pd.DataFrame, s: float, p_sib: float = 0.95, key: str = "akey") -> pd.DataFrame:
    """Sibling propagation. An unassigned record joins S1 X when a record with the same key is assigned to X with
    p >= p_sib and all such siblings agree on X. key = "akey" (addr_core + house number; then the name similarity,
    token_set, to the sibling must be >= s) or "name_core" (then the address similarity must be >= s)."""
    other = "name_core" if key == "akey" else "addr_core"
    anchors = rt[rt["selected"] & (rt["p"] >= p_sib) & (rt[key] != "")]
    one = anchors.groupby(key)["s1"].nunique()
    anchors = anchors[anchors[key].isin(one.index[one == 1])].drop_duplicates(key)[[key, "s1", other]]
    cand = rt[~rt["selected"] & (rt[key] != "")].reset_index().merge(anchors, on=key, suffixes=("", "_x"))
    cand["sim"] = [fuzz.token_set_ratio(a, b) / 100 for a, b in zip(cand[other], cand[f"{other}_x"])]
    return cand[cand["sim"] >= s]


def score_with(ev: Eval, add: pd.DataFrame) -> dict:
    """Corrected / raw F0.5 when the rescued (rec -> s1_x) links are added to the chainfix decision."""
    sel, tp, fp_raw, fp_w = ev.counts(T)
    n = len(ev.b_keys)
    i = np.searchsorted(ev.b_keys, add["s1_x"].to_numpy()).clip(max=n - 1)
    inb = ev.b_keys[i] == add["s1_x"].to_numpy()
    good = (add["true_key"].to_numpy() == add["s1_x"].to_numpy())
    tp = tp + np.bincount(i[inb & good], minlength=n)
    fp_raw = fp_raw + np.bincount(i[inb & ~good], minlength=n)
    fp_w = fp_w + np.bincount(i[inb & ~good], weights=add["w"].to_numpy()[inb & ~good], minlength=n)
    fn = ev.n_true - tp
    return {"f05": float(f05_per_s1(tp, fn, fp_raw).mean()), "f05_corrected": float(f05_per_s1(tp, fn, fp_w).mean()),
            "added": int(len(add)), "added_to_b": int(inb.sum()), "correct": int((inb & good).sum())}


def run_siblings() -> None:
    ev = load_chainfix_eval()
    log("chainfix eval loaded")
    res = {"signals": siblings(ev)}
    log(json.dumps(res["signals"]))
    rt = pd.read_parquet(CACHE / "pass10_records.parquet")
    base = score_with(ev, rescue(ev, rt, 2.0))  # s > 1 adds nothing
    res["base"] = base
    res["grid"] = {}
    for key, sims in (("akey", (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)), ("name_core", (0.0, 0.3, 0.5, 0.7, 0.9))):
        for p_sib in (0.95, 0.99):
            for s in sims:
                r = score_with(ev, rescue(ev, rt, s, p_sib, key))
                r["gain_corrected"] = r["f05_corrected"] - base["f05_corrected"]
                res["grid"][f"key={key}, p_sib={p_sib}, s={s}"] = r
                log(f"key={key} p_sib={p_sib} s={s}: {json.dumps(r)}")
    (CACHE / "pass10_siblings.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["loss", "structure", "siblings"])
    args = ap.parse_args()
    start_memory_guard()
    if args.cmd == "loss":
        res = loss_breakdown(load_chainfix_eval())
    elif args.cmd == "structure":
        res = structure()
    else:
        return run_siblings()
    (CACHE / f"pass10_{args.cmd}.json").write_text(json.dumps(res, indent=1))
    log(json.dumps(res))


if __name__ == "__main__":
    main()
