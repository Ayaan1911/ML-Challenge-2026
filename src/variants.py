"""Pass 8c: submission variants built from saved predictions, each in its own folder under output/variants/.
Never writes output/matching_results.tsv or data/cache/pred_test/ (the Pass 7 submission and its predictions).

Run: python -m src.variants threshold 0.85 t085     -> output/variants/t085/ (Pass 7 predictions, stricter t)
     python -m src.variants chainfix 0.80 chainfix  -> output/variants/chainfix/ (Pass 7 + src.chainfix delta)
Each folder gets matching_results.tsv (+ candidate_pairs.tsv when the candidate set changes), a zip holding
matching_results.tsv (the leaderboard upload), summary.json with change counts vs Pass 7, and validate.log.
"""
import argparse
import json
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CACHE, ROOT, id_key, peak_rss_mb, start_memory_guard
from src.norm_cache import norm_path
from src.postprocess import decide
from src.run_pipeline import write_grouped

VARIANTS = ROOT / "output" / "variants"
PASS7_T = 0.80
PASS7_CANDIDATES = ROOT / "output" / "candidate_pairs.tsv"
CHAINFIX_DELTA = CACHE / "pred_test_chainfix"  # written by src.chainfix test; Pass 7's pred_test is never touched
VALIDATOR = ("import sys, runpy; from src.data import start_memory_guard; start_memory_guard(); "
             "sys.argv = ['validate_submission.py'] + sys.argv[1:]; "
             "runpy.run_path('utils/validate_submission.py', run_name='__main__')")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


def load_pass7() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = pa.concat_tables(pq.read_table(p) for p in sorted((CACHE / "pred_test").glob("*.parquet")))
    return pred["s1"].to_numpy(), pred["rec"].to_numpy(), pred["prob"].to_numpy()


def with_delta(s1k: np.ndarray, rk: np.ndarray, prob: np.ndarray, delta: pd.DataFrame):
    """Replace every row of the delta's records by the delta rows (which hold all of those records' pairs)."""
    keep = ~np.isin(rk, delta["rec"].to_numpy())
    return (np.concatenate([s1k[keep], delta["s1"].to_numpy()]), np.concatenate([rk[keep], delta["rec"].to_numpy()]),
            np.concatenate([prob[keep], delta["prob"].to_numpy(np.float32)]))


def s1_country() -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(norm_path("test", "source1"), columns=["entity_id", "country_norm"])
    keys = id_key(t["entity_id"])
    order = np.argsort(keys)
    return keys[order], np.asarray(t["country_norm"].to_numpy(zero_copy_only=False), object)[order]


def change_counts(old: pd.Series, new: pd.Series, s1_all: np.ndarray, country: np.ndarray) -> dict:
    """old/new: chosen S1 key per matched record key. A pair is added / removed when that record -> S1 link is."""
    both = pd.concat([old.rename("old"), new.rename("new")], axis=1)
    added = both.loc[both["new"].notna() & (both["new"] != both["old"]), "new"].to_numpy(np.int64)
    removed = both.loc[both["old"].notna() & (both["old"] != both["new"]), "old"].to_numpy(np.int64)
    c_of = lambda s: country[np.searchsorted(s1_all, s)]  # noqa: E731
    out = {}
    for c in sorted(set(country)):
        n_old = int((c_of(old.to_numpy(np.int64)) == c).sum())
        n_add, n_rem = int((c_of(added) == c).sum()), int((c_of(removed) == c).sum())
        out[c] = {"pass7_pairs": n_old, "added": n_add, "removed": n_rem, "variant_pairs": n_old + n_add - n_rem}
    return out


def validate(folder, candidates, log_path=None) -> dict:
    """Organizer validator with and without --check-ids (matching file), check_candidates in both modes."""
    m, res, lines = folder / "matching_results.tsv", {}, []
    log_path = log_path or folder / "validate.log"
    runs = {"validator": [sys.executable, "-c", VALIDATOR, "--matching", str(m), "--candidate",
                          str(folder / "_no_candidate_file.tsv"), "--test-dir", "dataset/test"],
            "check_candidates": [sys.executable, "-m", "src.check_candidates", "--matching", str(m),
                                 "--candidate", str(candidates)]}
    for name, cmd in runs.items():
        for extra in ([], ["--check-ids"]):
            p = subprocess.run(cmd + extra, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok = p.returncode == 0 and "PASS" in p.stdout
            res[f"{name}{' --check-ids' if extra else ''}"] = "PASS" if ok else f"FAIL (exit {p.returncode})"
            lines += [f"=== {name} {' '.join(extra)} -> exit {p.returncode}", p.stdout, p.stderr]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"validation: {res}")
    if any(v != "PASS" for v in res.values()):
        raise SystemExit(f"{folder.name}: validation failed, see {log_path}")
    return res


def write_variant(name: str, t: float, delta: pd.DataFrame | None = None) -> dict:
    t0 = time.perf_counter()
    folder = VARIANTS / name
    folder.mkdir(parents=True, exist_ok=True)
    s1k, rk, prob = load_pass7()
    old_sel = decide(rk, prob, PASS7_T)
    old = pd.Series(s1k[old_sel], index=rk[old_sel])
    del old_sel
    if delta is not None:
        s1k, rk, prob = with_delta(s1k, rk, prob, delta)
    sel = decide(rk, prob, t)
    new = pd.Series(s1k[sel], index=rk[sel])
    s1_all, country = s1_country()
    if not np.isin(np.unique(s1k), s1_all).all():
        raise SystemExit("candidate S1 ids not in test source1")
    write_grouped(folder / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"], s1_all, s1k[sel], rk[sel])
    candidates = PASS7_CANDIDATES
    if delta is not None:  # the candidate set grew: write the variant's own candidate file
        candidates = folder / "candidate_pairs.tsv"
        write_grouped(candidates, ["source1_entity_id", "candidate_entity_ids"], s1_all, s1k, rk)
    del s1k, rk, prob, sel
    counts = change_counts(old, new, s1_all, country)
    log(f"{name}: {json.dumps(counts)}")
    res = validate(folder, candidates)
    zpath = folder / f"submission_{name}.zip"
    with zipfile.ZipFile(zpath.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(folder / "matching_results.tsv", "matching_results.tsv")
    zpath.with_suffix(".tmp").replace(zpath)
    summary = {"variant": name, "threshold": t, "chainfix": delta is not None, "changes_vs_pass7": counts,
               "matched_pairs": int(len(new)), "validation": res, "candidate_file": str(candidates.relative_to(ROOT)),
               "zip": str(zpath.relative_to(ROOT)), "zip_mb": round(zpath.stat().st_size / 2**20, 1),
               "runtime_min": round((time.perf_counter() - t0) / 60, 1), "peak_mb": round(peak_rss_mb())}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["threshold", "chainfix"])
    ap.add_argument("t", type=float)
    ap.add_argument("name")
    args = ap.parse_args()
    start_memory_guard()
    delta = None
    if args.kind == "chainfix":
        parts = sorted(CHAINFIX_DELTA.glob("*.parquet"))
        if not parts:
            raise SystemExit(f"no delta predictions in {CHAINFIX_DELTA}; run python -m src.chainfix test first")
        delta = pd.concat([pd.read_parquet(p, columns=["s1", "rec", "prob"]) for p in parts], ignore_index=True)
        log(f"chainfix delta: {len(delta):,} rows from {[p.stem for p in parts]}")
    write_variant(args.name, args.t, delta)


if __name__ == "__main__":
    main()
