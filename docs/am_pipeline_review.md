# Review of AM's parallel pipeline (origin/am-lead-pipeline)

Read, not imported or run, in Pass 5: `src/blocking.py` (4-channel blocker), `src/train.py` (classifier),
`src/postprocess.py` (inference and decision), `run_pipeline.py` (runner). Nothing below is implemented yet.
Benefit estimates replay AM's key functions (reimplemented for the analysis) on the **156,658 val true pairs that
our Pass 3 blocking + read-time pruning lose** (of 1,527,889; real-density val queries). "Recall points" are
shares of all val true pairs. They are upper bounds: our pruning keeps every exact-key hit, so a recovered pair
survives pruning, but the model still has to accept it, and every channel adds candidate volume.

## Worth porting (ranked)

| # | Idea (AM's blocker) | Lost val pairs it would recover (cap 20 per key) | Est. benefit | Cost / risk |
|---|---|---|---|---|
| 1 | **Address anchor keys**: first 1-5 digit number + first 3 characters of the raw name; postcode + first 3 characters of the name | 23,694 (15.1% of lost) | **+1.5 recall points** | Mean block ~1.5 S1 per key: cheap. Complements our `key_postcode_num` (postcode + first number), which misses records with no postcode; India addresses often lack a PIN |
| 2 | **Skip-bigram name keys**: pairs of name tokens at distance 1-2 among the first 4 tokens (stop-word list of common business words) | 20,360 (13.0%) | **+1.3 recall points** | Up to 5 keys per record, ~2 S1 per key: roughly +5 stored pairs per record before dedup. Catches the extra/missing-word cases our sorted-token key (all tokens equal) misses |
| 3 | **Compressed 8-character name prefix** (alphanumerics only, no spaces: "wal mart" = "walmart", "walmart supercenter" = "walmart") | 15,605 (10.0%) | **+1.0 recall point** | One key per record. Our `key_name_nospace` needs the whole name equal; the prefix tolerates suffix words |
| - | All three plus distinctive-token keys, union | 47,819 (30.5%) | **+3.1 recall points** (upper bound), roughly +2 F0.5 points if precision holds | Needs a blocking rerun (~20 min test, ~18 min val, ~40 min featurize/score) and a check of the pruned volume |

Distinctive single tokens (length >= 5, first 3 tokens) share a key with 53% of lost pairs but almost all those
keys exceed the cap of 20 (+0.18 points within the cap), so this channel adds little over our rare-word address
retriever.

## Not worth porting

- **Classifier** (`train.py`): 150 trees on 30k S1 with 1:3 hard negatives, and targets limited to the first 500k
  S2/S3 rows, so many true matches cannot be positives. Ours trains on 7.2M real-density pairs; nothing to take.
- **Decision rule** (`postprocess.py`): one candidate per S1 (the first in the list) and a fixed tau = 0.55. S1s
  have 3.46 matches on average (EDA), so top-1 per S1 caps recall near 1/3.5. Our rule (best S1 per record,
  threshold swept on corrected F0.5) keeps multi-match S1s.
- **RapidFuzz pre-screen** (skip scoring when name token_set_ratio < 40): on our 15.9M pruned val pairs it would
  skip 31.9% of pairs but drop 162,613 of 2,467,767 true pairs (6.6%), because many true pairs differ in name and
  are found by address. It would save about 10 of the 32 test-scoring minutes at a large recall cost. Not worth it.
- **Runner**: shells out with `python3`, `zip` and `os.system`, which fail on Windows and are not needed here.

## Also noted

- AM's blocker queries S1 against an S2/S3 index (the reverse of ours) and truncates each S1's candidate set to
  25 in arbitrary set order (`list(cand_set)[:25]`), so kept candidates are not the best ones.
- Everything is held in Python dicts of raw strings over the full test set, which will not fit next to anything
  else in 16 GB at 10M records.
