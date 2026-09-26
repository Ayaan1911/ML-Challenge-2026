"""Streaming version of utils/validate_submission.py's candidate_pairs.tsv checks.

The organizer validator keeps every candidate id in Python sets: with our ~47M candidate pairs it passes 7.5 GB
and cannot run on a 16 GB laptop. This applies the same rules line by line (header, one row per test S1, no
duplicate rows or ids within a list, only S2-/S3- ids, every matched id also a candidate; with --check-ids,
every id exists in test S2/S3), holding only the matching file in memory.
Run: python -m src.check_candidates [--check-ids]
"""
import argparse
import sys

from src.data import ROOT, peak_rss_mb, start_memory_guard


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        yield f.readline().rstrip("\n").split("\t")
        for line in f:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            yield s1, (rest.split(",") if rest.strip() else []) if tab else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=str(ROOT / "output" / "matching_results.tsv"))
    ap.add_argument("--candidate", default=str(ROOT / "output" / "candidate_pairs.tsv"))
    ap.add_argument("--test-dir", default=str(ROOT / "dataset" / "test"))
    ap.add_argument("--check-ids", action="store_true")
    args = ap.parse_args()
    start_memory_guard()
    with open(f"{args.test_dir}/test_source1.tsv", encoding="utf-8") as f:
        next(f)
        required = {line.split("\t", 1)[0].strip() for line in f if line.strip()}
    valid = None
    if args.check_ids:
        valid = set()
        for s in ("source2", "source3"):
            with open(f"{args.test_dir}/test_{s}.tsv", encoding="utf-8") as f:
                next(f)
                valid.update(line.split("\t", 1)[0].strip() for line in f if line.strip())
    rows = read_rows(args.matching)
    next(rows)
    matched = {s1: ids for s1, ids in rows if ids}
    errors, seen, n, pairs, not_cand = [], set(), 0, 0, 0
    rows = read_rows(args.candidate)
    if next(rows) != ["source1_entity_id", "candidate_entity_ids"]:
        errors.append("bad header")
    for s1, ids in rows:
        n += 1
        if ids is None:
            errors.append(f"malformed row {n}")
            continue
        if s1 in seen:
            errors.append(f"duplicate row {s1}")
        seen.add(s1)
        cand = set(ids)
        pairs += len(ids)
        if len(cand) != len(ids):
            errors.append(f"repeated id in list of {s1}")
        if any(not i.startswith(("S2-", "S3-")) for i in cand):
            errors.append(f"non S2/S3 id in list of {s1}")
        if valid is not None and not cand <= valid:
            errors.append(f"unknown id in list of {s1}")
        not_cand += len(set(matched.get(s1, ())) - cand)
    if seen != required:
        errors.append(f"S1 rows: {len(required - seen)} missing, {len(seen - required)} not in test")
    print(f"candidate_pairs.tsv: {n:,} rows, {pairs:,} candidate ids; matched ids not in candidates: {not_cand:,}; "
          f"id-existence check {'ON' if valid is not None else 'OFF'}; peak {peak_rss_mb():,.0f} MB")
    for e in errors[:20]:
        print("ERROR:", e)
    print("PASS" if not errors and not not_cand else f"FAIL ({len(errors)} errors)")
    return 1 if errors or not_cand else 0


if __name__ == "__main__":
    sys.exit(main())
