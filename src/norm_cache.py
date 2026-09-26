"""Normalize whole source files in parallel -> data/cache/norm_{split}_{source}.parquet (Table 5.2 columns).

Run:  python -m src.norm_cache dev            (dev slice only)
      python -m src.norm_cache train test     (all six files)
Each output stores the sha256 of src/normalize.py; a file is rebuilt only when normalize.py changed
(or with --force). So after merging AM's normalize.py, `python -m src.norm_cache train test` is all it takes.
"""
import argparse
import hashlib
import os
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CACHE, ROOT, peak_rss_mb, read_tsv, start_memory_guard, tsv_path

CHUNK = 50_000
WORKERS = max(1, (os.cpu_count() or 2) - 1)
SOURCES = ("source1", "source2", "source3")
SCHEMA = pa.schema([(k, pa.string()) for k in ("entity_id", "country_norm", "name_norm", "name_core",
                                                 "legal_suffix", "dba_name", "addr_norm", "addr_core",
                                                 "postcode")]
                   + [("numbers", pa.list_(pa.string())), ("landmark", pa.string())])
_META_KEY = b"normalize_sha256"


def norm_path(split: str, source: str):
    return CACHE / f"norm_{split}_{source}.parquet"


def normalize_sha() -> str:
    return hashlib.sha256((ROOT / "src" / "normalize.py").read_bytes()).hexdigest()


def is_fresh(split: str, source: str) -> bool:
    p = norm_path(split, source)
    return p.exists() and (pq.read_schema(p).metadata or {}).get(_META_KEY) == normalize_sha().encode()


def _work(cols: tuple[list, list, list, list]) -> tuple[pa.Table, int, float]:
    from src.normalize import normalize_record
    recs = [normalize_record({"entity_id": i, "business_name": n, "business_address": a, "country": c})
            for i, n, a, c in zip(*cols)]
    table = pa.Table.from_pylist(recs, schema=SCHEMA)
    return table, os.getpid(), peak_rss_mb()


def _source_frame(split: str, source: str):
    if split != "dev":
        return read_tsv(tsv_path(split, source))
    from src.dev_slice import load_dev
    df = read_tsv(tsv_path("train", source))
    return df[df["entity_id"].isin(load_dev(f"s{source[-1]}"))].reset_index(drop=True)


def build(split: str, source: str, pool: Pool) -> None:
    t0 = time.perf_counter()
    df = _source_frame(split, source)
    cols = [df[c].tolist() for c in ("entity_id", "business_name", "business_address", "country")]
    del df
    chunks = ((cols[0][i:i + CHUNK], cols[1][i:i + CHUNK], cols[2][i:i + CHUNK], cols[3][i:i + CHUNK])
              for i in range(0, len(cols[0]), CHUNK))
    out = norm_path(split, source)
    tmp = out.with_suffix(".tmp")
    schema = SCHEMA.with_metadata({_META_KEY: normalize_sha().encode()})
    worker_peak: dict[int, float] = {}
    with pq.ParquetWriter(tmp, schema) as w:
        for table, pid, peak in pool.imap(_work, chunks):
            w.write_table(table.replace_schema_metadata(schema.metadata))
            worker_peak[pid] = peak
    os.replace(tmp, out)  # never leave a half-written cache behind
    dt = time.perf_counter() - t0
    print(f"{out.name:32s} rows={len(cols[0]):>9,d} time={dt:6.1f}s rate={len(cols[0]) / dt:>9,.0f}/s "
          f"main_peak={peak_rss_mb():,.0f}MB worker_peak_max={max(worker_peak.values(), default=0):,.0f}MB "
          f"workers_sum={sum(worker_peak.values()):,.0f}MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+", choices=["dev", "train", "test"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS, help="fewer workers = less memory (~180 MB each)")
    args = ap.parse_args()
    start_memory_guard()
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for split in args.splits:
            for source in SOURCES:
                if not args.force and is_fresh(split, source):
                    print(f"{norm_path(split, source).name}: up to date, skipped")
                    continue
                build(split, source, pool)


if __name__ == "__main__":
    main()
