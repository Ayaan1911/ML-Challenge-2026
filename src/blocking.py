"""Blocking v2: for every S2/S3 query record retrieve candidate S1 records of the same country string.

Retrievers (union):
- exact keys (KeyIndex): name_core without spaces, sorted name_core tokens, postcode + first number,
  postcode alone, address word + postcode. Keys held by more than block_cap S1 records are skipped.
- name: char-3gram TF-IDF on name_core, prefix-filtered top-k, true cosine (tfidf_name_cos).
- address (addr_mode="rareword"): words of name_core + addr_core whose S1 document frequency is at most
  rare_df_frac of the country's S1; top-k S1 by summed IDF of shared rare words (addr_word_idf), then
  re-scored with the true char-3gram cosine of name_core + addr_core (tfidf_nameaddr_cos).
  addr_mode="trigram" is the Pass 2 retriever (prefix-filtered char-3gram top-k on name + address).
Pruning happens at READ time (read_cands(..., pruned=True) / prune()): per record keep every key hit
plus the top PRUNE["n"] S1 by prune_score = tfidf_name_cos + tfidf_nameaddr_cos + addr_word_idf / 20.
Blocking stores the unpruned candidates so pruning can be re-tuned on real-density validation without
re-blocking. The pruned set is what the model scores and what candidate_pairs.tsv must contain.

Output: data/cache/<name>/ with one parquet part per (country, chunk of query records), written as each
chunk finishes. A rerun skips finished parts (same config only). Progress + ETA: logs/blocking_<name>.log.
Run: python -m src.blocking {dev|train|test} [--name NAME] [--queries ids.txt] [--set key=value ...]
Needs the normalization cache: python -m src.norm_cache <split>
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

from src.data import CACHE, ROOT, peak_rss_mb
from src.norm_cache import norm_path

CONFIG = {
    "keys": ["key_name_nospace", "key_sorted_tokens", "key_postcode_num", "key_postcode", "key_word_postcode"],
    "block_cap": 50,         # skip exact keys shared by more S1 records than this
    "k_name": 5,             # top-k S1 per record from name TF-IDF
    "prefix_name": 3,        # name probe uses this many rarest trigrams (speed; see blocking_report.md)
    "addr_mode": "rareword",  # "rareword" (v2) or "trigram" (Pass 2)
    "k_addr": 5,             # top-k S1 per record from the address retriever
    "rare_df_frac": 0.002,   # rareword: a word is rare if held by <= this share of the country's S1
    "prefix_nameaddr": 8,    # trigram mode only
    "probe_mult": 3,         # trigram probes re-score probe_mult * k hits with the true cosine
    "min_cos": 0.3,          # cosine floor for the name / trigram retrievers
    "prune_n": None,         # None = store unpruned (default); pruning is applied at read time, see PRUNE
    "prune_floor": 0.0,
    "chunk": 200_000,        # query records per part file
    "n_features": 2 ** 20,   # hashing space (char 3-grams); words use 2**22
    "threads": max(1, (os.cpu_count() or 2) - 1),
}
PRUNE = {"n": 3, "floor": 0.0}  # tuned on real-density val queries, see docs/blocking_report.md
ALL_KEYS = ["key_name_nospace", "key_sorted_tokens", "key_postcode_num", "key_postcode", "key_word_postcode"]
SCORES = ["tfidf_name_cos", "tfidf_nameaddr_cos", "addr_word_idf"]
OUT_SCHEMA = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()), ("country", pa.string())]
                       + [(f, pa.int8()) for f in ALL_KEYS] + [(c, pa.float32()) for c in SCORES])
LOGS = ROOT / "logs"
_timings: dict[str, float] = {}


@contextmanager
def timed(stage: str):
    t = time.perf_counter()
    yield
    _timings[stage] = _timings.get(stage, 0.0) + time.perf_counter() - t


def load_norm(split: str, sources: tuple[str, ...]) -> pd.DataFrame:
    """Blocking columns from the normalization cache, plus the single-valued exact keys."""
    cols = ["entity_id", "country_norm", "name_core", "addr_core", "postcode", "numbers"]
    frames = []
    for s in sources:
        df = pq.read_table(norm_path(split, s), columns=cols).to_pandas()
        first = np.array([n[0] if len(n) else "" for n in df.pop("numbers")], dtype=object)
        pc = df["postcode"].to_numpy(object)
        df["key_postcode_num"] = np.where((pc != "") & (first != ""), pc + "|" + first, "")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.assign(key_name_nospace=df["name_core"].str.replace(" ", "", regex=False),
                     key_sorted_tokens=[" ".join(sorted(x.split())) for x in df["name_core"].tolist()],
                     key_postcode=df["postcode"])


def word_postcode_keys(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(key, row label) for every address word (>= 3 chars or a number, not the postcode) x postcode."""
    d = df.loc[df["postcode"] != "", ["addr_core", "postcode"]]
    w = d["addr_core"].str.split(" ").explode().dropna()
    pc = d["postcode"].loc[w.index].to_numpy(object)
    w_np = w.to_numpy(object)
    ok = ((w.str.len() >= 3) | w.str.isdigit()).to_numpy(bool) & (w_np != pc)
    return (w_np[ok] + "|" + pc[ok]).astype(object), w.index.to_numpy()[ok]


class KeyIndex:
    """Exact-key join from query keys to S1 owners. Keys held by more than `cap` S1 records are skipped."""

    def __init__(self, keys: np.ndarray, owners: np.ndarray, cap: int):
        d = pd.DataFrame({"k": keys, "o": owners})
        d = d[d["k"] != ""].drop_duplicates()
        codes, uniq = pd.factorize(d["k"].to_numpy(object))
        self.counts = np.bincount(codes, minlength=len(uniq))
        self.owners = d["o"].to_numpy()[np.argsort(codes, kind="stable")]
        self.starts = np.concatenate([[0], np.cumsum(self.counts)])[:-1]
        self.index, self.cap = pd.Index(uniq), cap

    def lookup(self, keys: np.ndarray, rec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All (rec, owner) pairs with equal keys, in the order of `rec`."""
        if not len(self.index):
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        c = self.index.get_indexer(keys)
        ok = c >= 0
        c, rec = c[ok], rec[ok]
        n = self.counts[c]
        c, rec, n = c[n <= self.cap], rec[n <= self.cap], n[n <= self.cap]
        within = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        return np.repeat(rec, n), self.owners[np.repeat(self.starts[c], n) + within]


def exact_pairs(s1_keys: np.ndarray, rec_keys: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    """(rec_idx, s1_idx) pairs with equal non-empty keys, skipping keys held by > cap S1. Sorted by rec_idx."""
    return KeyIndex(s1_keys, np.arange(len(s1_keys)), cap).lookup(rec_keys, np.arange(len(rec_keys)))


def _hash(hv: HashingVectorizer, texts: list[str], par: Parallel, threads: int):
    """HashingVectorizer is stateless and pure-Python per doc, so hash slices in parallel processes."""
    step = max(5_000, -(-len(texts) // threads))
    return sp.vstack(par(delayed(hv.transform)(texts[i:i + step]) for i in range(0, len(texts), step)),
                     format="csr")


def _top_per_row(r: np.ndarray, s: np.ndarray, v: np.ndarray, k: int):
    """Keep the k highest v per row r."""
    order = np.lexsort((-v, r))
    r, s, v = r[order], s[order], v[order]
    best = np.arange(len(r)) - np.searchsorted(r, r) < k
    return r[best], s[best], v[best]


class TfidfIndex:
    """char_wb 3-gram TF-IDF over S1 texts, top-k cosine lookup by prefix filtering.

    A full sparse product costs the sum of posting lengths of every query trigram, and common trigrams
    (city, state, "road", "ent") touch ~27k S1 per query at full scale (hours). So each query probes the
    index with only its `prefix` rarest trigrams (short postings, highest idf), keeps the best
    `probe_mult * k` S1 by that partial score, then re-scores them with the true full-vector cosine.
    """

    def __init__(self, s1_texts: list[str], cfg: dict, par: Parallel, prefix: int, probe: bool = True):
        self.cfg, self.par, self.prefix = cfg, par, prefix
        self.hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=cfg["n_features"],
                                    alternate_sign=False, norm=None, dtype=np.float32)
        x = _hash(self.hv, s1_texts, par, cfg["threads"])
        self.tt = TfidfTransformer(sublinear_tf=True).fit(x)
        df = np.bincount(x.indices, minlength=x.shape[1])
        self.rarity = np.where(df == 0, np.iinfo(np.int64).max, df)  # trigrams absent from S1 are useless probes
        self.a = self.tt.transform(x).astype(np.float32)  # S1 x F, unit rows, for re-scoring
        self.b = self.a.T.tocsr() if probe else None        # F x S1 postings, for probing

    def vectorize(self, texts: list[str]):
        return self.tt.transform(_hash(self.hv, texts, self.par, self.cfg["threads"])).astype(np.float32)

    def cosine(self, q, r: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.asarray(q[r].multiply(self.a[s]).sum(axis=1)).ravel().astype(np.float32)

    def _rarest(self, q):
        """Keep each row's `prefix` trigrams with the lowest (non-zero) S1 document frequency."""
        rows = np.repeat(np.arange(q.shape[0]), np.diff(q.indptr))
        order = np.lexsort((self.rarity[q.indices], rows))
        keep = order[np.arange(len(order)) - q.indptr[rows[order]] < self.prefix]
        return sp.csr_matrix((q.data[keep], (rows[keep], q.indices[keep])), shape=q.shape)

    def topk(self, q, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = sp_matmul_topn(self._rarest(q), self.b, top_n=k * self.cfg["probe_mult"],
                           n_threads=self.cfg["threads"]).tocoo()
        cos = self.cosine(q, c.row, c.col)
        ok = cos > self.cfg["min_cos"]
        return _top_per_row(c.row[ok], c.col[ok], cos[ok], k)


class RareWordIndex:
    """Top-k S1 by summed IDF of shared rare words. Posting lists are capped by construction, so a query
    costs at most (its words x rare_df_frac x n_S1) operations instead of a scan of common trigrams."""

    def __init__(self, s1_texts: list[str], cfg: dict, par: Parallel):
        self.cfg, self.par = cfg, par
        self.hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False, binary=True,
                                    n_features=2 ** 22, alternate_sign=False, norm=None, dtype=np.float32)
        x = _hash(self.hv, s1_texts, par, cfg["threads"])
        n1 = x.shape[0]
        df = np.bincount(x.indices, minlength=x.shape[1])
        rare = (df >= 1) & (df <= max(1.0, cfg["rare_df_frac"] * n1))
        self.idf = sp.diags(np.where(rare, np.log(n1 / np.maximum(df, 1)), 0).astype(np.float32))
        self.b = (x @ sp.diags(rare.astype(np.float32))).T.tocsr()  # V x S1 postings of rare words only

    def topk(self, texts: list[str], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = (_hash(self.hv, texts, self.par, self.cfg["threads"]) @ self.idf).tocsr()
        q.eliminate_zeros()
        c = sp_matmul_topn(q, self.b, top_n=k, n_threads=self.cfg["threads"]).tocoo()
        return c.row, c.col, c.data.astype(np.float32)


def prune_score(df: pd.DataFrame) -> np.ndarray:
    """Ranking score for pruning. max(cosines) alone buries true pairs found only by the rare-word
    retriever (low trigram cosine); adding the rare-word IDF keeps them (val: 89.9% vs 85.2% at N=3)."""
    return (df["tfidf_name_cos"].to_numpy(np.float32) + df["tfidf_nameaddr_cos"].to_numpy(np.float32)
            + df["addr_word_idf"].to_numpy(np.float32) / 20)


def prune(df: pd.DataFrame, n=None, floor=None) -> pd.DataFrame:
    """Read-time pruning of a candidate frame (all pairs of a record must be in df)."""
    n = PRUNE["n"] if n is None else n
    floor = PRUNE["floor"] if floor is None else floor
    rec = pd.factorize(df["cand_id"])[0]
    has_key = (df[ALL_KEYS].to_numpy() > 0).any(axis=1)
    return df[prune_mask(rec, has_key, prune_score(df), n, floor)]


def prune_mask(rec: np.ndarray, has_key: np.ndarray, score: np.ndarray, n, floor: float) -> np.ndarray:
    """Keep every key hit plus, per record, the top-n candidates by score that reach the floor."""
    if n is None:
        return np.ones(len(rec), dtype=bool)
    order = np.lexsort((-score, rec))
    rank = np.empty(len(rec), dtype=np.int64)
    rank[order] = np.arange(len(rec)) - np.searchsorted(rec[order], rec[order])
    return has_key | ((rank < n) & (score >= floor))


class CountryBlocker:
    """All S1-side indexes of one country; block() handles one chunk of query records."""

    def __init__(self, s1: pd.DataFrame, cfg: dict, par: Parallel):
        self.cfg, self.n1 = cfg, len(s1)
        self.s1_ids = pa.array(s1["entity_id"].tolist())
        own = np.arange(len(s1))
        with timed("s1_keys"):
            self.keys = {f: KeyIndex(s1[f].to_numpy(object), own, cfg["block_cap"])
                         for f in cfg["keys"] if f != "key_word_postcode"}
            if "key_word_postcode" in cfg["keys"]:
                self.keys["key_word_postcode"] = KeyIndex(*word_postcode_keys(s1), cfg["block_cap"])
        with timed("s1_index"):
            nameaddr = (s1["name_core"] + " " + s1["addr_core"]).str.strip().tolist()
            trigram = cfg["addr_mode"] == "trigram"
            self.name = TfidfIndex(s1["name_core"].tolist(), cfg, par, cfg["prefix_name"])
            self.nameaddr = TfidfIndex(nameaddr, cfg, par, cfg["prefix_nameaddr"], probe=trigram)
            self.words = None if trigram else RareWordIndex(nameaddr, cfg, par)

    def block(self, rec: pd.DataFrame) -> pd.DataFrame:
        """rec: one chunk of query records (RangeIndex from 0). Returns deduplicated, pruned pairs."""
        cfg, n1, parts = self.cfg, self.n1, []
        ridx = np.arange(len(rec))
        with timed("keys"):
            for f, index in self.keys.items():
                if f == "key_word_postcode":
                    keys, labels = word_postcode_keys(rec)
                else:
                    keys, labels = rec[f].to_numpy(object), ridx
                r, s = index.lookup(keys, labels)
                parts.append((r, s, f, np.ones(len(r), dtype=np.float32)))
        with timed("name_tfidf"):
            r, s, v = self.name.topk(self.name.vectorize(rec["name_core"].tolist()), cfg["k_name"])
            parts.append((r, s, "tfidf_name_cos", v))
        with timed("addr_retriever"):
            texts = (rec["name_core"] + " " + rec["addr_core"]).str.strip().tolist()
            q = self.nameaddr.vectorize(texts)
            if self.words is None:
                r, s, v = self.nameaddr.topk(q, cfg["k_addr"])
            else:
                r, s, idf = self.words.topk(texts, cfg["k_addr"])
                parts.append((r, s, "addr_word_idf", idf))
                v = self.nameaddr.cosine(q, r, s)
            parts.append((r, s, "tfidf_nameaddr_cos", v))
        with timed("dedup_prune"):
            return self._merge(parts, rec)

    def _merge(self, parts: list, rec: pd.DataFrame) -> pd.DataFrame:
        n1 = self.n1
        key = np.concatenate([p[0].astype(np.int64) * n1 + p[1] for p in parts])
        g = pd.DataFrame({c: np.concatenate([p[3] if p[2] == c else np.zeros(len(p[0]), np.float32)
                                             for p in parts]) for c in ALL_KEYS + SCORES})
        g = g.groupby(key, sort=False).max()
        r, s = g.index.to_numpy() // n1, g.index.to_numpy() % n1
        keep = prune_mask(r, (g[ALL_KEYS].to_numpy() > 0).any(axis=1), prune_score(g),
                          self.cfg["prune_n"], self.cfg["prune_floor"])
        out = g[keep].reset_index(drop=True)
        out[ALL_KEYS] = out[ALL_KEYS].astype(np.int8)
        out.insert(0, "cand_id", rec["entity_id"].to_numpy(object)[r[keep]])
        out.insert(0, "s1_id", self.s1_ids.take(pa.array(s[keep])).to_numpy(zero_copy_only=False))
        return out


def _slug(country: str) -> str:
    """File-safe, collision-free name for an arbitrary country string (country is an open set)."""
    return re.sub(r"[^a-z0-9]+", "_", country)[:30] + "-" + hashlib.md5(country.encode()).hexdigest()[:6]


class Progress:
    def __init__(self, name: str, total_records: int):
        LOGS.mkdir(exist_ok=True)
        self.path, self.total, self.done, self.t0 = LOGS / f"blocking_{name}.log", total_records, 0, time.time()

    def log(self, msg: str, records: int = 0) -> None:
        self.done += records
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done) if self.done else float("nan")
        line = (f"{time.strftime('%H:%M:%S')} [{self.done:,}/{self.total:,} records, {el / 60:.1f} min, "
                f"ETA {eta / 60:.1f} min, peak {peak_rss_mb():,.0f} MB] {msg}")
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _prepare_outdir(outdir: Path, cfg: dict, fresh: bool) -> None:
    meta = outdir / "_config.json"  # leading "_": parquet readers skip it
    want = {k: v for k, v in cfg.items() if k != "threads"}
    if fresh and outdir.exists():
        shutil.rmtree(outdir)
    if meta.exists() and json.loads(meta.read_text()) != want:
        raise SystemExit(f"{outdir} was built with a different config; rerun with --fresh")
    outdir.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(want, indent=1))


def run(split: str, cfg: dict, name: str, queries: set | None = None, sample_records: int = 0,
        fresh: bool = False) -> dict:
    t0 = time.perf_counter()
    outdir = CACHE / name
    _prepare_outdir(outdir, cfg, fresh)
    with timed("load"):
        s1 = load_norm(split, ("source1",))
        rec = load_norm(split, ("source2", "source3"))
    n_rec_all = len(rec)
    if queries is not None:
        rec = rec[rec["entity_id"].isin(queries)]
    if sample_records:
        rec = rec.sample(n=min(sample_records, len(rec)), random_state=42)
    by_country = {c: g.reset_index(drop=True) for c, g in rec.groupby("country_norm", sort=True)}
    s1_by_country = {c: g.reset_index(drop=True) for c, g in s1.groupby("country_norm", sort=True)}
    prog = Progress(name, sum(len(g) for c, g in by_country.items() if c in s1_by_country))
    prog.log(f"start split={split} S1={len(s1):,} queries={len(rec):,} config={json.dumps(cfg)}")
    with Parallel(n_jobs=cfg["threads"]) as par:
        for country, rec_c in by_country.items():
            if country not in s1_by_country:
                continue  # no S1 with this country string -> no candidates (EDA: pairs share the country)
            chunks = [(i, lo) for i, lo in enumerate(range(0, len(rec_c), cfg["chunk"]))]
            todo = [(i, lo) for i, lo in chunks if not (outdir / f"{_slug(country)}-{i:04d}.parquet").exists()]
            prog.log(f"{country}: {len(chunks) - len(todo)} of {len(chunks)} chunks already done",
                     records=sum(min(cfg["chunk"], len(rec_c) - lo) for i, lo in chunks if (i, lo) not in todo))
            if not todo:
                continue
            blocker = CountryBlocker(s1_by_country[country], cfg, par)
            prog.log(f"{country}: S1 index built ({len(s1_by_country[country]):,} S1)")
            for i, lo in todo:
                part = rec_c.iloc[lo:lo + cfg["chunk"]].reset_index(drop=True)
                out = blocker.block(part).assign(country=country)
                path = outdir / f"{_slug(country)}-{i:04d}.parquet"
                with timed("write"):
                    pq.write_table(pa.Table.from_pandas(out, schema=OUT_SCHEMA, preserve_index=False),
                                   path.with_suffix(".tmp"))
                    os.replace(path.with_suffix(".tmp"), path)
                prog.log(f"{country} chunk {i + 1}/{len(chunks)}: {len(out):,} pairs", records=len(part))
    pairs = sum(pq.ParquetFile(p).metadata.num_rows for p in outdir.glob("*.parquet"))
    stats = {"split": split, "name": name, "n_s1": len(s1), "n_rec": len(rec), "n_rec_all": n_rec_all,
             "pairs": pairs, "runtime_s": round(time.perf_counter() - t0, 1), "peak_mb": round(peak_rss_mb()),
             "stages_s": {k: round(v, 1) for k, v in _timings.items()}, "out": str(outdir)}
    prog.log(f"done: {json.dumps(stats)}")
    return stats


def read_cands(name: str, columns: list[str] | None = None, pruned: bool = False) -> pd.DataFrame:
    """All parts of a blocking output. pruned=True applies prune() part by part (parts hold whole records)."""
    if not pruned:
        return pd.read_parquet(CACHE / name, columns=columns)
    parts = [prune(pd.read_parquet(p)) for p in sorted((CACHE / name).glob("*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    return df if columns is None else df[columns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "train", "test"], help="which normalized S1/S2/S3 to use")
    ap.add_argument("--name", help="output folder under data/cache (default cands_<split>)")
    ap.add_argument("--queries", help="text file of S2/S3 ids to query (default: all records of the split)")
    ap.add_argument("--sample-records", type=int, default=0, help="calibration: random sample of query records")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON", help='config overrides, e.g. k_addr=10')
    ap.add_argument("--fresh", action="store_true", help="delete earlier parts of this output first")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown config key {k}")
        cfg[k] = json.loads(v)
    queries = set(Path(args.queries).read_text(encoding="utf-8").split()) if args.queries else None
    stats = run(args.split, cfg, args.name or f"cands_{args.split}", queries, args.sample_records, args.fresh)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
