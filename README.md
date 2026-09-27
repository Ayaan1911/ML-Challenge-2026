# Business Entity Resolution — team neural forge

Links every Source 1 business record (the hub, `S1`) to the Source 2 / Source 3 records (`S2`, `S3`) that describe
the same business. The pipeline has four steps:
- **Normalization:** rule-based cleaning of names and addresses (multilingual and noise-hardened).
- **Blocking:** a multi-channel candidate search, pruned to about 4.7 candidate S1 per test record.
- **Pair scoring:** a LightGBM classifier over 60 pair, blocking and context features.
- **Decision:** each record goes to its single best S1 if its probability is at least t.

It writes the two files the challenge asks for, `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
The methodology is in `Documentation_template.md` at the root of the submission zip.

**No external data, APIs or lookups are used.** Everything is computed from the challenge TSVs with the open-source
libraries listed below. There are no pretrained models, no web services and no geocoding.

## Environment

- **Python:** 3.11 (developed with 3.11.1 on Windows 11). Linux and macOS work too; paths are handled with `pathlib`.
- **Hardware:** 16 GB RAM laptop (Intel i5-12500H, no GPU). Every command stays below about 6.5 GB.
- **Setup:**

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q               # 71 tests, about 15 s
```

`requirements.txt` pins the exact versions of the development environment.

| package | version | license |
|---|---|---|
| lightgbm | 4.7.0 | MIT |
| rapidfuzz | 3.14.6 | MIT |
| sparse_dot_topn | 1.2.0 | Apache-2.0 |
| anyascii | 0.3.3 | ISC |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-3-Clause |
| pandas | 3.0.6 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| pyarrow | 25.0.1 | Apache-2.0 |
| joblib | 1.6.0 | BSD-3-Clause |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| narwhals | 2.26.0 | MIT |
| python-dateutil | 2.9.0.post0 | Apache-2.0 or BSD-3-Clause (dual) |
| six | 1.17.0 | MIT |
| tzdata | 2026.4 | Apache-2.0 |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| pytest | 9.1.1 | MIT (tests only) |
| pluggy / iniconfig | 1.6.0 / 2.3.0 | MIT (pytest dependencies) |
| packaging | 26.3 | Apache-2.0 or BSD-2-Clause |
| Pygments / colorama | 2.21.0 / 0.4.6 | BSD-2-Clause / BSD-3-Clause (pytest dependencies) |

## Data layout

Put the challenge files here, unchanged:

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

Intermediate files go to `data/cache/` (about 10 GB for a full run), split lists to `data/splits/` and outputs to
`output/`. Every stage writes to a `.tmp` file and renames it at the end, so a stopped run never leaves a
half-written file.

Each command starts an in-process memory guard. It exits cleanly with code 3 when free system memory drops below
1.5 GB; rerunning the command resumes where it stopped.

## Full reproduction (train from scratch)

Run from this folder, in order. The times and peak memory were measured on the hardware above with the final
configuration.

| # | command | what it does | time | peak memory |
|---|---|---|---|---|
| 1 | `python -m src.data` | Parses the 7 TSVs, checks row counts and ids, caches them as parquet | about 2 min | about 3 GB |
| 2 | `python -m src.split` | 80/20 train/val split of train S1 ids (seed 42, stratified by has-match x country) -> `data/splits/` | about 30 s | about 2 GB |
| 3 | `python -m src.norm_cache train test --workers 6` | Normalizes all six source files -> `data/cache/norm_*.parquet` | 2.7 min | 2.9 GB + 1.2 GB in workers |
| 4 | `python -m src.candsets val` | Val blocking: val records plus a 20% sample of the other train records, queried against the full train S1 index and labelled | 28.9 min | 6.5 GB |
| 5 | `python -m src.valsplit` | Splits the pruned val candidates by query record into a train side (val S1 half A) and an eval side (half B) | 2.7 min | 1.5 GB |
| 6 | `python -m src.matcher features val` | 60 features for the 15.97M pruned val pairs | 12.7 min | 4.0 GB |
| 7 | `python -m src.matcher train` | LightGBM on the train side -> `models/lgbm.txt` | 7.7 min | 4.2 GB |
| 8 | `python -m src.postprocess sweep` | Scores the eval side, picks t on corrected F0.5 -> `models/threshold.json` | about 8 min (1 min when predictions are cached) | 3.4 GB |
| 9 | `python -m src.postprocess tune` | Optional: decision-rule study -> `models/decision.json` (Pass 8 adopted no extra rule: t alone) | 4.4 min | 3.5 GB |
| 10 | `python -m src.blocking test` | Test blocking: 142.2M stored pairs, 46.8M after pruning | 36.9 min | 5.3 GB |
| 11 | `python -m src.run_pipeline test` | Features, model and decision over the pruned test candidates -> `output/matching_results.tsv`, `output/candidate_pairs.tsv` | 123.8 min | 4.0 GB |
| 12 | variant step (below) | Only for the chain-name variants | 2.6 + 6.9 + 6.5 min | 4.1 GB |
| 13 | validation (below) | Organizer validator and the streamed candidate check | about 2 min | 2.4 GB |

About 4.1 hours in total; 4.0 hours without the optional step 9.

**Determinism.**
- Steps 1-6 and 8-13 are deterministic: seeds are fixed, and ties break by stable sorts.
- LightGBM training (step 7) is seeded, but multithreaded histogram construction can change the last bits.
  `models/lgbm.txt` is the exact model behind the submitted files.
- Run the fast path to reproduce the outputs byte for byte.

### Split lists (not shipped; regenerated deterministically)

`data/splits/` is rebuilt by step 2 (`train_s1_ids.txt`, `val_s1_ids.txt`) and step 5 (`valsplit_*`). With the
challenge data and the pinned library versions, the files match these SHA-256 values. Hashes are of the LF form;
a Git checkout with `core.autocrlf` may add CRs.

```
9b29105042a85f427e70e2ef03195d0e05163eb585747181a8488e35043dc2b7  data/splits/train_s1_ids.txt
a646eddcd70f431149be3c069efe75968f0e38c8503bb60157577c54d01c8ac8  data/splits/val_s1_ids.txt
```

## Fast path: test outputs from the included model

`models/` holds the trained model (`lgbm.txt`), the chosen threshold (`threshold.json`) and the decision rules
(`decision.json`). To reproduce the submitted test outputs without training:

```
python -m src.data                                 # optional integrity check (about 2 min)
python -m src.norm_cache test --workers 6          # about 1.5 min
python -m src.blocking test                        # 36.9 min, 5.3 GB
<!-- VARIANT COMMANDS -->
```

This takes about 2.7 hours. `src.run_pipeline test` is resumable: per-part predictions in `data/cache/pred_test/`
are reused while the model, `features.py` and `normalize.py` are unchanged.

### Variant in this package

The submitted files were produced by the commands above (variant `{{FINAL_VARIANT}}`, decision threshold
{{FINAL_THRESHOLD}}, chain-name candidate step: {{FINAL_CHAINFIX}}). The other variants were built the same way:

| variant | commands after step 10 | writes |
|---|---|---|
| pass7 | `python -m src.run_pipeline test` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| t085 / t090 | `run_pipeline test`, then `python -m src.variants threshold 0.85 t085` (or `0.90 t090`) | `output/variants/t0xx/matching_results.tsv`; candidates from `run_pipeline` |
| chainfix | `run_pipeline test`, `python -m src.chainfix test france`, `python -m src.chainfix test india`, `python -m src.variants chainfix 0.80 chainfix` | `output/variants/chainfix/` (both files) |
| chainfix_t085 | same, last command `python -m src.variants chainfix 0.85 chainfix_t085` | `output/variants/chainfix_t085/` |

The chain-name candidate step (`src/chainfix.py`) targets S1s left with no candidate. It adds every record of the
same country with the identical normalized core name and first house number, computes their blocking columns
with the blocking code itself, and re-scores only the records involved. It aborts unless recomputed blocking
columns match stored ones and untouched control records reproduce their saved probabilities exactly.
`python -m src.chainfix eval` measures it on the eval side: corrected F0.5 0.9531 -> 0.9561.

## Validation

```
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/_none.tsv --test-dir dataset/test --check-ids
python -m src.check_candidates --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --check-ids
```

`utils/validate_submission.py` is the organizer validator. It holds all 47M candidate ids in Python sets (more than
7.5 GB), so on a 16 GB machine the candidate file is checked by `src.check_candidates` instead. That tool applies
the same rules, streamed: 1.2 GB, or 2.1 GB with `--check-ids`.

## Code map

| module | role |
|---|---|
| `src/data.py` | TSV parsing and parquet cache, lossless int64 id keys, memory guard |
| `src/split.py`, `src/valsplit.py` | train/val S1 split; train/eval halves of the val candidate set |
| `src/normalize.py`, `src/norm_cache.py` | NormRecord rules (names, legal forms, addresses, postcodes, numbers, landmarks; anyascii transliteration) and the parallel cache builder |
| `src/blocking.py`, `src/candsets.py` | candidate channels, prefix-filtered TF-IDF top-k, pruning; labelled val candidate sets |
| `src/features.py`, `src/matcher.py` | pair features; blocking and context features, LightGBM training and scoring |
| `src/postprocess.py`, `src/metric.py` | decision layer (best S1 per record, threshold, Pass 8 rule study), competition F0.5 |
| `src/run_pipeline.py` | end-to-end test run and TSV writers |
| `src/chainfix.py`, `src/variants.py` | chain-name candidate step and submission variants from saved predictions |
| `src/check_candidates.py` | streamed validator for `candidate_pairs.tsv` |
| `src/dev_slice.py` | dev-slice ids (only for `norm_cache dev`) |
| `tests/` | unit tests: normalization, features, blocking keys, decision rules, metric, TSV writer |
