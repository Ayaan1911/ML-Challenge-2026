"""Blocking v1: for every S2/S3 record retrieve candidate S1 records of the same country string.

Retrievers (union): exact keys on name_core without spaces / sorted name_core tokens / postcode+first
number (keys shared by > block_cap S1 records are skipped), plus char-3gram TF-IDF top-k on name_core
and on name_core + addr_core (sparse_dot_topn, cosine > min_cos).
Output: data/cache/cands_{split}.parquet, one row per (s1_id, cand_id).
Run: python -m src.blocking {dev|train|test} [--k 5] [--out path] [--sample-records N]
Needs the normalization cache: python -m src.norm_cache <split>
"""
import argparse
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from src.data import CACHE, peak_rss_mb
from src.norm_cache import norm_path

CONFIG = {
    "k_name": 5,             # top-k S1 per record from name TF-IDF (k sweep: docs/blocking_report.md)
    "k_nameaddr": 5,         # top-k S1 per record from name+address TF-IDF
    "block_cap": 50,         # skip exact keys shared by more S1 records than this
    "min_cos": 0.3,          # TF-IDF cosine floor
    "prefix_name": 4,        # probe with this many rarest trigrams of the record name (speed)
    "prefix_nameaddr": 8,    # ... of the record name + address
    "probe_mult": 3,         # re-score probe_mult * k probe hits with the true cosine
    "chunk": 200_000,        # records per matmul / dedup chunk
    "n_features": 2 ** 20,   # hashing space
    "threads": max(1, (os.cpu_count() or 2) - 1),
    "seed": 42,
}
FLAGS = ["key_name_nospace", "key_sorted_tokens", "key_postcode_num"]
COSINES = ["tfidf_name_cos", "tfidf_nameaddr_cos"]
OUT_SCHEMA = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()), ("country", pa.string())]
                       + [(f, pa.int8()) for f in FLAGS] + [(c, pa.float32()) for c in COSINES])
_timings: dict[str, float] = {}


@contextmanager
def timed(stage: str):
    t = time.perf_counter()
    yield
    _timings[stage] = _timings.get(stage, 0.0) + time.perf_counter() - t


def load_norm(split: str, sources: tuple[str, ...]) -> pd.DataFrame:
    """Blocking columns from the normalization cache, plus the three exact keys."""
    frames = []
    for s in sources:
        t = pq.read_table(norm_path(split, s),
                          columns=["entity_id", "country_norm", "name_core", "addr_core", "postcode", "numbers"])
        first = pc.binary_join(pc.list_slice(t["numbers"], 0, 1), "")
        ok = pc.and_(pc.not_equal(t["postcode"], ""), pc.not_equal(first, ""))
        key_pc = pc.if_else(ok, pc.binary_join_element_wise(t["postcode"], first, "|"), "")
        t = t.drop_columns(["numbers", "postcode"]).append_column("key_postcode_num", key_pc)
        frames.append(t.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    core = df["name_core"].tolist()
    return df.assign(key_name_nospace=df["name_core"].str.replace(" ", "", regex=False),
                     key_sorted_tokens=[" ".join(sorted(x.split())) for x in core])


def exact_pairs(s1_keys: np.ndarray, rec_keys: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """All (rec_idx, s1_idx) with equal non-empty keys, skipping keys held by > cap S1. Sorted by rec_idx."""
    codes, _ = pd.factorize(np.concatenate([s1_keys, rec_keys]))
    n1 = len(s1_keys)
    s1c, rc = codes[:n1], codes[n1:].copy()
    s1c = np.where(s1_keys == "", -1, s1c)
    rc[rec_keys == ""] = -1
    valid = np.nonzero(s1c >= 0)[0]
    order = valid[np.argsort(s1c[valid], kind="stable")]
    counts = np.bincount(s1c[valid], minlength=codes.max() + 1)
    starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
    rec_idx = np.nonzero(rc >= 0)[0]
    n = counts[rc[rec_idx]]
    keep = (n >= 1) & (n <= cap)
    rec_idx, n = rec_idx[keep], n[keep]
    rep_rec = np.repeat(rec_idx, n)
    within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    return rep_rec, order[np.repeat(starts[rc[rec_idx]], n) + within]


class TfidfIndex:
    """char_wb 3-gram TF-IDF over S1 texts, top-k cosine lookup by prefix filtering.

    A full sparse product costs the sum of posting lengths of every query trigram, and common trigrams
    (city, state, "road", "ent") touch ~27k S1 per query at full scale (hours). So each query probes the
    index with only its `prefix` rarest trigrams (short postings, highest idf), keeps the best
    `probe_mult * k` S1 by that partial score, then re-scores them with the true full-vector cosine.
    """

    def __init__(self, s1_texts: list[str], cfg: dict, par: Parallel, prefix: int):
        self.cfg, self.par, self.prefix = cfg, par, prefix
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=cfg["n_features"],
                                    alternate_sign=False, norm=None, dtype=np.float32)
        x = self._hash(s1_texts)
        self.tt = TfidfTransformer(sublinear_tf=True).fit(x)
        df = np.bincount(x.indices, minlength=x.shape[1])
        self.rarity = np.where(df == 0, np.iinfo(np.int64).max, df)  # trigrams absent from S1 are useless probes
        self.a = self.tt.transform(x).astype(np.float32)  # S1 x F, unit rows, for re-scoring
        self.b = self.a.T.tocsr()                           # F x S1 postings, for probing

    def _hash(self, texts: list[str]):
        """HashingVectorizer is stateless and pure-Python per doc, so hash slices in parallel processes."""
        step = max(5_000, -(-len(texts) // self.cfg["threads"]))
        return sp.vstack(self.par(delayed(self.hv.transform)(texts[i:i + step])
                                  for i in range(0, len(texts), step)), format="csr")

    def vectorize(self, texts: list[str]):
        return self.tt.transform(self._hash(texts)).astype(np.float32)

    def _rarest(self, q):
        """Keep each row's `prefix` trigrams with the lowest (non-zero) S1 document frequency."""
        rows = np.repeat(np.arange(q.shape[0]), np.diff(q.indptr))
        order = np.lexsort((self.rarity[q.indices], rows))
        keep = order[np.arange(len(order)) - q.indptr[rows[order]] < self.prefix]
        return sp.csr_matrix((q.data[keep], (rows[keep], q.indices[keep])), shape=q.shape)

    def topk(self, q, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = sp_matmul_topn(self._rarest(q), self.b, top_n=k * self.cfg["probe_mult"],
                           n_threads=self.cfg["threads"]).tocoo()
        r, s = c.row, c.col
        cos = np.asarray(q[r].multiply(self.a[s]).sum(axis=1)).ravel().astype(np.float32)
        ok = cos > self.cfg["min_cos"]
        r, s, cos = r[ok], s[ok], cos[ok]
        order = np.lexsort((-cos, r))
        r, s, cos = r[order], s[order], cos[order]
        first = np.searchsorted(r, r)  # index of each row's first entry (r is sorted)
        best = np.arange(len(r)) - first < k
        return r[best], s[best], cos[best]


def _part(rec: np.ndarray, s1: np.ndarray, n1: int, col: str, val) -> pd.DataFrame:
    d = {"key": rec.astype(np.int64) * n1 + s1}
    for c in FLAGS + COSINES:
        d[c] = val if c == col else np.zeros(len(rec), dtype=np.float32)
    return pd.DataFrame(d)


def block_country(s1: pd.DataFrame, rec: pd.DataFrame, country: str, cfg: dict, writer, par: Parallel) -> int:
    n1 = len(s1)
    with timed("exact_keys"):
        exact = {f: exact_pairs(s1[f].to_numpy(object), rec[f].to_numpy(object), cfg["block_cap"]) for f in FLAGS}
    with timed("s1_index"):
        nameaddr = (s1["name_core"] + " " + s1["addr_core"]).str.strip().tolist()
        idx = {"tfidf_name_cos": TfidfIndex(s1["name_core"].tolist(), cfg, par, cfg["prefix_name"]),
               "tfidf_nameaddr_cos": TfidfIndex(nameaddr, cfg, par, cfg["prefix_nameaddr"])}
    texts = {"tfidf_name_cos": lambda d: d["name_core"].tolist(),  # built per chunk to bound memory
             "tfidf_nameaddr_cos": lambda d: (d["name_core"] + " " + d["addr_core"]).str.strip().tolist()}
    ks = {"tfidf_name_cos": cfg["k_name"], "tfidf_nameaddr_cos": cfg["k_nameaddr"]}
    s1_ids, rec_ids = pa.array(s1["entity_id"].tolist()), pa.array(rec["entity_id"].tolist())
    total = 0
    for lo in range(0, len(rec), cfg["chunk"]):
        hi = min(lo + cfg["chunk"], len(rec))
        parts = []
        for f, (r, s) in exact.items():
            a, b = np.searchsorted(r, [lo, hi])
            parts.append(_part(r[a:b], s[a:b], n1, f, np.ones(b - a, dtype=np.float32)))
        for col, index in idx.items():
            with timed(f"vec_{col}"):
                q = index.vectorize(texts[col](rec.iloc[lo:hi]))
            with timed(f"topk_{col}"):
                r, s, v = index.topk(q, ks[col])
            parts.append(_part(r + lo, s, n1, col, v))
        with timed("dedup_write"):
            g = pd.concat(parts, ignore_index=True).groupby("key", sort=False).max()
            keys = g.index.to_numpy()
            cols = {"s1_id": s1_ids.take(pa.array(keys % n1)), "cand_id": rec_ids.take(pa.array(keys // n1)),
                    "country": pa.array(np.full(len(g), country, dtype=object), pa.string())}
            cols.update({f: pa.array(g[f].to_numpy().astype(np.int8)) for f in FLAGS})
            cols.update({c: pa.array(g[c].to_numpy(np.float32)) for c in COSINES})
            writer.write_table(pa.table(cols, schema=OUT_SCHEMA))
        total += len(g)
    return total


def run(split: str, cfg: dict, out, sample_records: int = 0) -> dict:
    t0 = time.perf_counter()
    with timed("load"):
        s1 = load_norm(split, ("source1",))
        rec = load_norm(split, ("source2", "source3"))
    n_rec_all = len(rec)
    if sample_records:
        rec = rec.sample(n=min(sample_records, len(rec)), random_state=cfg["seed"]).reset_index(drop=True)
    total = 0
    tmp = out.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, OUT_SCHEMA) as w, Parallel(n_jobs=cfg["threads"]) as par:
        for country, s1_c in s1.groupby("country_norm", sort=True):
            rec_c = rec[rec["country_norm"] == country].reset_index(drop=True)
            if len(rec_c):
                total += block_country(s1_c.reset_index(drop=True), rec_c, country, cfg, w, par)
    os.replace(tmp, out)
    return {"split": split, "k": cfg["k_name"], "prefix": [cfg["prefix_name"], cfg["prefix_nameaddr"]],
            "n_s1": len(s1), "n_rec": len(rec), "n_rec_all": n_rec_all,
            "pairs": total, "runtime_s": round(time.perf_counter() - t0, 1), "peak_mb": round(peak_rss_mb()),
            "stages_s": {k: round(v, 1) for k, v in _timings.items()}, "out": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "train", "test"])
    ap.add_argument("--k", type=int, help="sets both k_name and k_nameaddr")
    ap.add_argument("--prefix", type=int, nargs=2, metavar=("NAME", "NAMEADDR"))
    ap.add_argument("--out")
    ap.add_argument("--sample-records", type=int, default=0, help="calibration: random sample of S2/S3 records")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    if args.k:
        cfg.update(k_name=args.k, k_nameaddr=args.k)
    if args.prefix:
        cfg.update(prefix_name=args.prefix[0], prefix_nameaddr=args.prefix[1])
    out = CACHE / (args.out or f"cands_{args.split}.parquet")
    print(json.dumps(run(args.split, cfg, out, args.sample_records)), flush=True)


if __name__ == "__main__":
    main()
