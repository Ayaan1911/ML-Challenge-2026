"""Pass 7 overnight driver: every stage resumable, a commit after each stage, submission replaced only when validated.

Stages run as separate processes (each starts the memory guard and exits with code 3 below 1,500 MB available).
A stage that stops for memory is retried after memory recovers (up to 3 times); finished stages are recorded in
data/cache/pass7_state.json and skipped on a rerun. Run: python -m src.pass7   (rerun the same command to resume)
"""
import json
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.blocking import CONFIG, PRUNE, always_key_hit, prune_mask, prune_score
from src.data import CACHE, ROOT, available_mb, load_truth
from src.norm_cache import norm_path
from src.split import load_split

STATE = CACHE / "pass7_state.json"
STAGED = ROOT / "output" / "pass7"
REPORT = ROOT / "docs" / "pass7_run.md"
PASS5_F05_CORRECTED = 0.9302158352270148  # models/threshold.json of Pass 5 (commit 4e7a3aa)
PY = sys.executable


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(ROOT / "logs" / "pass7_driver.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def mark(stage: str, **info) -> None:
    s = state()
    s[stage] = {"done": True, **info}
    STATE.write_text(json.dumps(s, indent=1))


def git_commit(paths: list[str], msg: str) -> None:
    subprocess.run(["git", "add", "--", *paths], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "-q", "-m", f"{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                       cwd=ROOT, check=True)
        log(f"committed: {msg}")


def sh(args: list[str], logname: str) -> None:
    """Run one command; on a memory-guard stop (exit 3) wait for memory and retry."""
    for attempt in range(4):
        t0 = time.time()
        with open(ROOT / "logs" / f"{logname}.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%H:%M:%S')} attempt {attempt + 1}: {' '.join(args)}\n")
            f.flush()
            code = subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"{' '.join(args)} -> exit {code} after {(time.time() - t0) / 60:.1f} min")
        if code == 0:
            return
        if code != 3 or attempt == 3:
            raise SystemExit(f"stage failed: {' '.join(args)} (exit {code}), see logs/{logname}.log")
        for _ in range(60):  # up to 30 min for memory to come back
            if available_mb() > 4_000:
                break
            time.sleep(30)
        log(f"memory guard stop; retrying with {available_mb():,.0f} MB available")


def stage(name: str, fn) -> None:
    if state().get(name, {}).get("done"):
        log(f"[{name}] already done, skipped")
        return
    log(f"[{name}] start")
    t0 = time.time()
    extra = fn() or {}
    mark(name, minutes=round((time.time() - t0) / 60, 1), **extra)
    log(f"[{name}] done in {(time.time() - t0) / 60:.1f} min")


def clear_if_config_changed(outdir) -> None:
    """Blocking refuses to mix parts built with another config: remove stale parts before a fresh build."""
    want = {k: v for k, v in CONFIG.items() if k != "threads"}
    meta = outdir / "_config.json"
    if meta.exists() and json.loads(meta.read_text()) != want:
        shutil.rmtree(outdir)
        log(f"removed {outdir.name}: built with an older blocking config")


# ---------- stages ----------

def s_norm():
    sh(["-m", "src.norm_cache", "train", "test", "--workers", "6"], "pass7_norm_cache")


def s_block_val():
    clear_if_config_changed(CACHE / "cands_val_parts")
    sh(["-m", "src.candsets", "val"], "pass7_candsets_val")


def s_block_report():
    """Real-density val pair recall (unpruned / pruned at several depths) and the F0.5 ceiling."""
    from src.valsplit import part_row_groups
    truth = load_truth()
    val = load_split("val")
    s1c = pq.read_table(norm_path("train", "source1"), columns=["entity_id", "country_norm"]).to_pandas()
    ctry = s1c.set_index("entity_id")["country_norm"]
    n_true = pd.Series({s: len(truth[s]) for s in val})
    f = pq.ParquetFile(CACHE / "cands_val.parquet")
    found = {n: [] for n in ("unpruned", 2, 3, 4, 5)}
    vol = {n: 0 for n in found}
    nrec = 0
    for groups in part_row_groups(f):
        t = f.read_row_groups(groups).to_pandas()
        rec = pd.factorize(t["cand_id"])[0]
        nrec += rec.max() + 1
        key, score = always_key_hit(t), prune_score(t)
        for n in found:
            m = np.ones(len(t), bool) if n == "unpruned" else prune_mask(rec, key, score, n, PRUNE["floor"])
            vol[n] += int(m.sum())
            hit = t[m & (t["label"] == 1) & (t["query_group"] == "target")]
            found[n].append(hit["s1_id"].value_counts())
    rows = []
    for n, parts in found.items():
        tp = pd.concat(parts).groupby(level=0).sum().reindex(n_true.index, fill_value=0)
        f05 = np.where(n_true == 0, 1.0, 1.25 * tp / (1.25 * tp + 0.25 * (n_true - tp)).where(n_true > 0, 1))
        by_c = tp.groupby(ctry.reindex(tp.index)).sum() / n_true.groupby(ctry.reindex(n_true.index)).sum()
        rows.append({"candidates": f"pruned n={n}" if n != "unpruned" else "unpruned", "pair recall": tp.sum() / n_true.sum(),
                     **{f"recall {c}": v for c, v in by_c.items()}, "F0.5 ceiling": f05.mean(),
                     "pairs per query record": vol[n] / nrec})
    df = pd.DataFrame(rows)
    lines = ["# Pass 7 blocking at real density (generated by `python -m src.pass7`)", "",
             f"Val queries against the full 2.2M train S1 index, {int(n_true.sum()):,} true val pairs. Pass 3/5 "
             f"reference (same queries): pair recall 0.9019 unpruned, **0.8975 pruned (n=3)**, F0.5 ceiling 0.9476, "
             f"4.8 pairs per query record. Pruning keeps every hit of the Pass 3 keys plus the top n by prune_score; "
             f"the Pass 7 channels (address anchors, skip-bigrams, 8-character prefix) compete for those n slots, "
             f"with true TF-IDF cosines recomputed for every pair. PRUNE n = {PRUNE['n']} is used downstream.", "",
             "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    (ROOT / "docs" / "blocking_pass7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    git_commit(["docs/blocking_pass7.md"], "wip(pass7): real-density val blocking recall")
    r = df.set_index("candidates").loc[f"pruned n={PRUNE['n']}"]
    return {"pair_recall_pruned": round(float(r["pair recall"]), 4), "f05_ceiling": round(float(r["F0.5 ceiling"]), 4)}


def s_valsplit():
    sh(["-m", "src.valsplit"], "pass7_valsplit")


def s_features():
    sh(["-m", "src.matcher", "features", "val"], "pass7_features_val")


def s_train():
    sh(["-m", "src.matcher", "train"], "pass7_train")


def s_sweep():
    sh(["-m", "src.postprocess", "sweep"], "pass7_sweep")
    th = json.loads((ROOT / "models" / "threshold.json").read_text())
    keep = th["f05_corrected"] >= PASS5_F05_CORRECTED
    git_commit(["docs/matcher_report.md", "docs/feature_importance.md"] + (["models"] if keep else []),
               "wip(pass7): retrained LightGBM and threshold sweep" + ("" if keep else " (model NOT kept)"))
    return {"f05_corrected": th["f05_corrected"], "f05": th["f05"], "t": th["t"], "kept": keep}


def s_errors():
    sh(["-m", "src.postprocess", "errors"], "pass7_errors")
    git_commit(["docs/errors_val.csv"], "wip(pass7): errors_val.csv")


def s_block_test():
    clear_if_config_changed(CACHE / "cands_test")
    sh(["-m", "src.blocking", "test"], "pass7_blocking_test")


def s_test():
    sh(["-m", "src.run_pipeline", "test", "--out-dir", str(STAGED)], "pass7_run_pipeline")


def s_validate():
    m, c = STAGED / "matching_results.tsv", STAGED / "candidate_pairs.tsv"
    runner = ("import sys, runpy; from src.data import start_memory_guard; start_memory_guard(); "
              "sys.argv = ['validate_submission.py'] + sys.argv[1:]; "
              "runpy.run_path('utils/validate_submission.py', run_name='__main__')")
    base = ["--matching", str(m), "--candidate", str(STAGED / "_no_candidate_file.tsv"), "--test-dir", "dataset/test"]
    sh(["-c", runner, *base], "pass7_validate")
    sh(["-c", runner, *base, "--check-ids"], "pass7_validate")
    sh(["-m", "src.check_candidates", "--matching", str(m), "--candidate", str(c)], "pass7_validate")
    sh(["-m", "src.check_candidates", "--matching", str(m), "--candidate", str(c), "--check-ids"], "pass7_validate")
    out = ROOT / "output"
    for name in ("matching_results.tsv", "candidate_pairs.tsv", "run_summary.json"):  # validated: replace
        shutil.copyfile(STAGED / name, out / f"{name}.tmp")
        (out / f"{name}.tmp").replace(out / name)
    with zipfile.ZipFile(out / "submission_pass7.zip.tmp", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "matching_results.tsv", "matching_results.tsv")
    (out / "submission_pass7.zip.tmp").replace(out / "submission_pass7.zip")
    log("output/matching_results.tsv, candidate_pairs.tsv and submission_pass7.zip written")


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    log("pass 7 driver start")
    stage("norm_cache", s_norm)
    stage("blocking_val", s_block_val)
    stage("blocking_report", s_block_report)
    stage("valsplit", s_valsplit)
    stage("features_val", s_features)
    stage("train", s_train)
    stage("sweep", s_sweep)
    if not state()["sweep"]["kept"]:
        log(f"STOP: corrected F0.5 {state()['sweep']['f05_corrected']:.4f} < Pass 5 {PASS5_F05_CORRECTED:.4f}; "
            "Pass 5 submission kept, no test run")
        return
    stage("errors", s_errors)
    stage("blocking_test", s_block_test)
    stage("run_pipeline_test", s_test)
    stage("validate_and_publish", s_validate)
    REPORT.write_text("# Pass 7 run (generated by `python -m src.pass7`)\n\n| stage | minutes | details |\n|---|---|---|\n"
                      + "\n".join(f"| {k} | {v.get('minutes', '')} | "
                                  f"{ {x: y for x, y in v.items() if x not in ('done', 'minutes')} or ''} |"
                                  for k, v in state().items()) + "\n", encoding="utf-8")
    log("pass 7 driver finished")


if __name__ == "__main__":
    main()
