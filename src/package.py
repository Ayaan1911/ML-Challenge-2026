"""Pass 9: final submission package (team "neural forge"). Repo tooling; not shipped in the package itself.

Run: python -m src.package build --variant pass7 [--leaderboard 0.8917]
         -> dist/neural_forge_submission/ (layout of docs/ORGANIZER_README.md), validated, zipped as
            dist/neural_forge_submission.zip and dist/neural_forge_submission_no_candidates.zip
     python -m src.package smoke
         -> copies the packaged code to a temp folder, creates a fresh venv from its requirements.txt, runs pytest
            and the fast path on a ~5,000 test-S1 slice with the included model, and compares the result with the
            same rows of the Pass 7 output (needs a prior build and the Pass 7 caches)
Nothing here runs a full-scale stage: build copies validated outputs, and the smoke test scores a slice.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CACHE, DATASET, ROOT, id_key, key_id, peak_rss_mb, start_memory_guard

TEAM = "neural forge"
DIST = ROOT / "dist"
PKG = DIST / "neural_forge_submission"
CODE = PKG / "code" / "business_entity_resolution"
OUT = ROOT / "output"

# modules needed to reproduce both output files from the raw data (analysis and driver scripts stay in the repo)
SHIP_SRC = ["__init__", "data", "split", "dev_slice", "normalize", "norm_cache", "blocking", "candsets", "valsplit",
            "features", "matcher", "postprocess", "run_pipeline", "metric", "check_candidates", "chainfix", "variants"]
SHIP_MODELS = ["lgbm.txt", "threshold.json", "decision.json", "train_summary.json"]
SHIP_OTHER = ["requirements.txt", "pytest.ini", "utils/validate_submission.py"]

# variant -> validated source files, eval corrected F0.5 (docs/pass8c_variants.md), leaderboard (None = unknown),
# and the commands that produce it after the common pipeline (README "Variant" block)
VARIANTS = {
    "pass7": dict(matching=OUT / "pass7" / "matching_results.tsv", candidates=OUT / "pass7" / "candidate_pairs.tsv",
                  eval=0.9531, lb=0.8917, t=0.80, chainfix=False,
                  cmds=["python -m src.run_pipeline test   # -> output/matching_results.tsv, output/candidate_pairs.tsv"]),
    "chainfix": dict(matching=OUT / "variants" / "chainfix" / "matching_results.tsv",
                     candidates=OUT / "variants" / "chainfix" / "candidate_pairs.tsv", eval=0.9561, lb=None, t=0.80,
                     chainfix=True, cmds=["python -m src.run_pipeline test",
                                          "python -m src.chainfix test france",
                                          "python -m src.chainfix test india",
                                          "python -m src.variants chainfix 0.80 chainfix   # -> output/variants/chainfix/"]),
    "chainfix_t085": dict(matching=OUT / "variants" / "chainfix_t085" / "matching_results.tsv",
                          candidates=OUT / "variants" / "chainfix_t085" / "candidate_pairs.tsv", eval=0.9558, lb=None,
                          t=0.85, chainfix=True, cmds=["python -m src.run_pipeline test",
                                                       "python -m src.chainfix test france",
                                                       "python -m src.chainfix test india",
                                                       "python -m src.variants chainfix 0.85 chainfix_t085   # -> output/variants/chainfix_t085/"]),
    "t085": dict(matching=OUT / "variants" / "t085" / "matching_results.tsv", candidates=OUT / "pass7" / "candidate_pairs.tsv",
                 eval=0.9528, lb=None, t=0.85, chainfix=False,
                 cmds=["python -m src.run_pipeline test   # candidate_pairs.tsv comes from here",
                       "python -m src.variants threshold 0.85 t085   # -> output/variants/t085/matching_results.tsv"]),
    "t090": dict(matching=OUT / "variants" / "t090" / "matching_results.tsv", candidates=OUT / "pass7" / "candidate_pairs.tsv",
                 eval=0.9518, lb=None, t=0.90, chainfix=False,
                 cmds=["python -m src.run_pipeline test   # candidate_pairs.tsv comes from here",
                       "python -m src.variants threshold 0.90 t090   # -> output/variants/t090/matching_results.tsv"]),
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [peak {peak_rss_mb():,.0f} MB] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # binary copy: models stay byte-exact
    a, b = sha256(src), sha256(dst)
    if a != b:
        raise SystemExit(f"SHA-256 mismatch after copying {src} -> {dst}")
    return a


def fill(text: str, variant: str, leaderboard: float | None) -> str:
    v = VARIANTS[variant]
    lb = leaderboard if leaderboard is not None else v["lb"]
    block = "\n".join(v["cmds"])  # the marker sits inside the README's fast-path code block
    subs = {"{{FINAL_VARIANT}}": variant, "{{FINAL_EVAL}}": f"{v['eval']:.4f}",
            "{{FINAL_LEADERBOARD}}": f"{lb:.4f}".rstrip("0") if lb is not None else "[fill in: final leaderboard F0.5]",
            "{{FINAL_THRESHOLD}}": f"{v['t']:.2f}", "{{FINAL_CHAINFIX}}": "yes" if v["chainfix"] else "no",
            "{{SUBMISSION_DATE}}": time.strftime("%Y-%m-%d"), "<!-- VARIANT COMMANDS -->": block,
            "{{CHAINFIX_LEADERBOARD}}": f"{lb:.4f}".rstrip("0") if variant == "chainfix" and lb is not None else "[fill in if uploaded]"}
    for k, s in subs.items():
        text = text.replace(k, s)
    return text


def build(variant: str, leaderboard: float | None) -> None:
    t0 = time.perf_counter()
    v = VARIANTS[variant]
    for f in (v["matching"], v["candidates"]):
        if not f.exists():
            raise SystemExit(f"{variant}: validated source {f} is missing")
    if PKG.exists():
        shutil.rmtree(PKG)
    sums = {}
    sums["output/matching_results.tsv"] = copy_verified(v["matching"], PKG / "output" / "matching_results.tsv")
    sums["output/candidate_pairs.tsv"] = copy_verified(v["candidates"], PKG / "output" / "candidate_pairs.tsv")
    log(f"outputs copied from {v['matching'].parent.relative_to(ROOT)} (SHA-256 verified)")
    for m in SHIP_SRC:
        copy_verified(ROOT / "src" / f"{m}.py", CODE / "src" / f"{m}.py")
    for t in sorted((ROOT / "tests").glob("*.py")):
        copy_verified(t, CODE / "tests" / t.name)
    for m in SHIP_MODELS:
        sums[f"code/business_entity_resolution/models/{m}"] = copy_verified(ROOT / "models" / m, CODE / "models" / m)
    for f in SHIP_OTHER:
        copy_verified(ROOT / f, CODE / f)
    (CODE / "README.md").write_text(fill((ROOT / "README.md").read_text(encoding="utf-8"), variant, leaderboard),
                                    encoding="utf-8")
    doc = fill((ROOT / "Documentation_template.md").read_text(encoding="utf-8"), variant, leaderboard)
    (PKG / "Documentation_template.md").write_text(doc, encoding="utf-8")
    (PKG / "SHA256SUMS.txt").write_text("".join(f"{h}  {p}\n" for p, h in sums.items()), encoding="utf-8")
    log(f"code, models, README and methodology written for variant {variant}")

    from src.variants import validate  # organizer validator + check_candidates on the packaged files
    res = validate(PKG / "output", PKG / "output" / "candidate_pairs.tsv", DIST / f"validate_{variant}.log")

    zips = {}
    for name, skip in (("neural_forge_submission.zip", None),
                       ("neural_forge_submission_no_candidates.zip", "output/candidate_pairs.tsv")):
        z = DIST / name
        with zipfile.ZipFile(z.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in sorted(p for p in PKG.rglob("*") if p.is_file()):
                arc = f.relative_to(PKG).as_posix()
                if arc != skip:
                    zf.write(f, arc)
        z.with_suffix(".tmp").replace(z)
        zips[name] = round(z.stat().st_size / 2**20, 1)
    files = {p.relative_to(PKG).as_posix(): round(p.stat().st_size / 2**20, 2) for p in sorted(PKG.rglob("*")) if p.is_file()}
    summary = {"variant": variant, "sha256": sums, "validation": res, "zips_mb": zips,
               "package_mb": round(sum(files.values()), 1), "files_mb": files,
               "runtime_min": round((time.perf_counter() - t0) / 60, 1), "peak_mb": round(peak_rss_mb())}
    (DIST / f"package_{variant}.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: summary[k] for k in ("variant", "validation", "zips_mb", "package_mb", "runtime_min")}))


# ---------- reproducibility smoke test ----------

SMOKE = DIST / "smoke"
N_SMOKE = 5000


def filter_lines(src: Path, dst: Path, ids: set[str]) -> int:
    """Header plus every line whose first field is in ids, byte for byte."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src, "rb") as f, open(dst, "wb") as g:
        g.write(f.readline())
        for line in f:
            if line.split(b"\t", 1)[0].decode() in ids:
                g.write(line)
                n += 1
    return n


def build_slice(root: Path) -> set[str]:
    """~5,000 random test S1, every record with a pruned Pass 7 candidate among them (whole records, unpruned rows),
    S1 context from the full candidate set on the pruned rows, the raw TSV lines of every entity involved, and the
    expected output rows. Returns the chosen S1 ids."""
    from src.matcher import CTX_S1, part_country, part_paths, read_part, s1_context
    s1_all = pq.read_table(CACHE / "test_source1.parquet", columns=["entity_id"])["entity_id"].to_numpy(zero_copy_only=False)
    chosen = np.random.default_rng(42).choice(s1_all, N_SMOKE, replace=False)
    ck = id_key(chosen.tolist())
    pred = pa.concat_tables(pq.read_table(p, columns=["s1", "rec"]) for p in sorted((CACHE / "pred_test").glob("*.parquet")))
    recs = np.unique(pred["rec"].to_numpy()[np.isin(pred["s1"].to_numpy(), ck)])
    del pred
    out = root / "data" / "cache" / "cands_test"
    out.mkdir(parents=True, exist_ok=True)
    s1_needed = set(chosen.tolist())
    by_country: dict[str, list] = {}
    for p in part_paths("test"):
        by_country.setdefault(part_country(p), []).append(p)
    for country, paths in by_country.items():
        ctx = s1_context("test", paths)  # full-set S1 context, row-aligned with read_part
        for p, c in zip(paths, ctx):
            pruned = read_part("test", p)
            m = np.isin(id_key(pruned["cand_id"].to_numpy(object)), recs)
            keep = pd.concat([pruned.loc[m, ["s1_id", "cand_id"]].reset_index(drop=True), c[m].reset_index(drop=True)], axis=1)
            raw = pd.read_parquet(p)
            raw = raw[np.isin(id_key(raw["cand_id"].to_numpy(object)), recs)]
            if raw.empty:
                continue
            raw = raw.merge(keep, on=["s1_id", "cand_id"], how="left")  # NaN context = pruned away on read
            raw[CTX_S1] = raw[CTX_S1].astype(np.float32)
            pq.write_table(pa.Table.from_pandas(raw, preserve_index=False), out / p.name)
            s1_needed |= set(raw["s1_id"].tolist())
        log(f"slice {country}: parts written")
    rec_ids = set(key_id(recs).to_pylist())
    for s, ids in (("source1", s1_needed), ("source2", rec_ids), ("source3", rec_ids)):
        n = filter_lines(DATASET / "test" / f"test_{s}.tsv", root / "dataset" / "test" / f"test_{s}.tsv", ids)
        log(f"slice test_{s}.tsv: {n:,} lines")
    want = set(chosen.tolist())
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        filter_lines(OUT / "pass7" / name, root / f"expected_{name}", want)
    return want


def smoke() -> None:
    t0 = time.perf_counter()
    if not CODE.exists():
        raise SystemExit("run python -m src.package build first")
    if SMOKE.exists():
        shutil.rmtree(SMOKE)
    work = SMOKE / "business_entity_resolution"
    shutil.copytree(CODE, work)
    chosen = build_slice(work)
    for f in ("expected_matching_results.tsv", "expected_candidate_pairs.tsv"):
        shutil.move(work / f, SMOKE / f)
    log(f"slice built in {(time.perf_counter() - t0) / 60:.1f} min")
    base = Path(getattr(sys, "_base_executable", sys.executable))
    steps = [([str(base), "-m", "venv", str(SMOKE / "venv")], ROOT)]
    py = str(SMOKE / "venv" / ("Scripts" if os.name == "nt" else "bin") / "python")
    steps += [([py, "-m", "pip", "install", "-q", "-r", "requirements.txt"], work),
              ([py, "-m", "pytest", "-q", "-p", "no:cacheprovider"], work),
              ([py, "-m", "src.norm_cache", "test", "--workers", "2"], work),
              ([py, "-m", "src.run_pipeline", "test", "--out-dir", "output"], work)]
    results = []
    for cmd, cwd in steps:
        t = time.perf_counter()
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        name = " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))
        results.append({"step": name, "exit": p.returncode, "min": round((time.perf_counter() - t) / 60, 1),
                        "tail": (p.stdout + p.stderr).strip().splitlines()[-3:]})
        log(f"{name} -> exit {p.returncode} ({results[-1]['min']} min)")
        if p.returncode != 0:
            (SMOKE / "failed_step.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
            if "pytest" not in name:  # report a test failure but still run the pipeline
                raise SystemExit(f"smoke step failed: {name}, see {SMOKE / 'failed_step.log'}")
    compare = {}
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        got = [ln for ln in (work / "output" / name).read_bytes().splitlines()[1:] if ln.split(b"\t", 1)[0].decode() in chosen]
        want = (SMOKE / f"expected_{name}").read_bytes().splitlines()[1:]
        compare[name] = {"rows": len(want), "identical": got == want,
                         "non_empty": sum(1 for ln in want if not ln.endswith(b"\t"))}
    summary = {"s1": len(chosen), "steps": results, "compare": compare,
               "runtime_min": round((time.perf_counter() - t0) / 60, 1), "peak_mb": round(peak_rss_mb())}
    (DIST / "smoke.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(compare))
    if not all(c["identical"] for c in compare.values()):
        raise SystemExit("smoke test: packaged code does NOT reproduce the Pass 7 rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "smoke"])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="pass7")
    ap.add_argument("--leaderboard", type=float, help="final leaderboard F0.5, written into the methodology")
    args = ap.parse_args()
    start_memory_guard()
    build(args.variant, args.leaderboard) if args.cmd == "build" else smoke()


if __name__ == "__main__":
    main()
