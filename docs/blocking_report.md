# Blocking report (Pass 3)

Hand-written from measured runs on 25-26 Sept (a full `python -m src.eval_blocking` rerun was skipped
to save ~20 min; the script regenerates this file with the same sections).

**Read the real-density numbers first.** The dev slice (44k S1) is far less dense than the real data
(1.7-2.2M S1 compete for every record), and it overstated recall by about 6-10 points. Dev tables are
kept below only for relative comparisons between variants and are **superseded** by the real-density
validation figures.

## Chosen configuration

`CONFIG` / `PRUNE` in `src/blocking.py`:

- Exact keys (block cap 50): name_core without spaces, sorted name_core tokens, postcode + first
  number, postcode alone, address word + postcode.
- Name retriever: char_wb 3-gram TF-IDF on name_core, top-5 by prefix filtering (probe with the 3
  rarest trigrams, re-score 15 hits with the true cosine, cosine > 0.3).
- Address retriever (new, "rareword"): words of name_core + addr_core held by at most 0.2% of the
  country's S1; top-5 S1 by summed IDF of shared rare words (`addr_word_idf`), re-scored with the true
  char-3gram cosine of name_core + addr_core (`tfidf_nameaddr_cos`).
- Blocking stores **unpruned** candidates. **Read-time pruning** (`prune()`): per query record keep
  every key hit plus the top 3 S1 by `prune_score = tfidf_name_cos + tfidf_nameaddr_cos +
  addr_word_idf / 20`. The pruned set is what the model scores and what goes into
  `candidate_pairs.tsv`.

## Real-density validation (authoritative)

Every record whose true S1 is one of the 441,365 val S1 (1,527,889 true pairs) was queried against the
full 2.2M train S1 index, together with a random 20% of all other records.

| candidates | pair recall | India | US | F0.5 ceiling | pairs per query record |
|---|---|---|---|---|---|
| unpruned | 0.9019 | 0.8454 | 0.9397 | 0.9495 | 10.2 |
| **pruned (read time)** | **0.8975** | **0.8408** | **0.9354** | **0.9476** | **4.8** |

Pruning sweep at real density (per query record, over all its candidates):

| ranking | prune_n | pair recall | pairs per record |
|---|---|---|---|
| none (unpruned) | - | 0.9019 | 10.2 |
| max_cos | 2 | 0.8436 | 3.9 |
| max_cos | 3 | 0.8516 | 4.8 |
| max_cos | 4 | 0.8567 | 5.8 |
| max_cos | 6 | 0.9013 | 7.5 |
| prune_score | 2 | 0.8953 | 3.9 |
| **prune_score** | **3** | **0.8975** | **4.8** |
| prune_score | 4 | 0.8989 | 5.8 |
| prune_score | 6 | 0.9012 | 7.6 |

`max_cos` = max(tfidf_name_cos, tfidf_nameaddr_cos) was the first pruning rule tried. It buries true
pairs that only the rare-word retriever finds (their trigram cosine is modest) and costs 4.5 points at
N=3-4. `prune_score` adds the rare-word IDF and keeps 99.5% of unpruned recall at 47% of the pairs.

Recall of each retriever alone at real density (100,000 random val-matched records, unpruned):
address rare-word retriever 0.853, sorted-token key 0.409, name-without-spaces key 0.392, **name
TF-IDF 0.328** (0.73 on dev: at real density its top 5 fill up with look-alike names), postcode keys
~0.05 each, postcode + first number 0.012. Union 0.903.

## Full runs with the chosen config

| run | query records | S1 index | pairs stored (unpruned) | pairs after pruning | positives | runtime | peak memory |
|---|---|---|---|---|---|---|---|
| **test** (`data/cache/cands_test/`) | 9,969,589 | 1,732,544 test S1 | 99,370,911 (10.0 / record) | ~48M (~4.8 / record, val rate) | - | 22.2 min (33.5 min on the first run) | 5.2 GB |
| **val** (`data/cache/cands_val.parquet`) | 3,286,602 | 2,206,821 train S1 | 33,299,743 | 15,861,208 | 2,479,889 unpruned (7.4%), 2,467,767 kept (15.6%) | 17.7 min blocking + ~8 min labelling | 5.4 GB |
| **trainsample** (`data/cache/cands_trainsample.parquet`) | 6,154,802 (70%) | 2,206,821 train S1 | pending | pending | pending | ~24 min blocking for 28 of 32 parts (5.47M of 6.15M records) before the interruption; resumable | 4.7 GB |

The test figure after pruning is an estimate from the val pruning rate; the exact count comes out when
the pipeline writes `candidate_pairs.tsv`.

## How the sampled validation set biases F0.5, and the correction

Validation queries are every record whose true S1 is a val S1, plus a random 20% of all other records.
Every true pair of a val S1 is therefore candidate-eligible, so recall is measured exactly. False
positives are not: a record that belongs to a train S1 (or to nobody) can still be predicted for a val
S1, and only 20% of those records were queried. The other 80% would have added about four times as
many such false positives, so raw validation precision, and with it F0.5, is **overestimated**. The
effect is largest on singleton val S1s, which lose their free 1.0 on any false positive.

Correction: each pair from a sampled "other" record carries `weight = 5` (= 1 / 0.2) in
`cands_val.parquet`. When scoring, count a false positive from such a record as 5 false positives
(expected-value correction): precision = TP / (TP + FP_target + 5 * FP_other). A singleton val S1
already scores 0 on any false positive, so the correction only changes S1s that have matches. It cannot
recover the singletons that the unqueried 80% would have hit, so the corrected score is still slightly
optimistic on singletons. Report raw and corrected F0.5 side by side.

## Dev slice (superseded; relative comparisons only)

44,136 S1, 1,169,459 records, 153,439 true pairs. All variants stored unpruned; "p" = name prefix.

| variant | pair recall | India | US | F0.5 ceiling | pairs |
|---|---|---|---|---|---|
| Pass 2 (trigram address retriever, 3 keys, p4) | 0.9605 | 0.9281 | 0.9821 | 0.9858 | 5.42M |
| Pass 2 + postcode keys | 0.9614 | 0.9282 | 0.9836 | 0.9861 | 5.44M |
| rare-word df 0.0005, k5, p4 | 0.9441 | 0.9105 | 0.9665 | 0.9752 | 8.01M |
| rare-word df 0.001, k5, p4 | 0.9545 | 0.9241 | 0.9747 | 0.9805 | 8.69M |
| rare-word df 0.002, k5, p4 | 0.9660 | 0.9384 | 0.9844 | 0.9859 | 9.05M |
| rare-word df 0.001, k10, p4 | 0.9629 | 0.9357 | 0.9809 | 0.9844 | 12.79M |
| **chosen: rare-word df 0.002, k5, p3** | 0.9627 | 0.9342 | 0.9816 | 0.9845 | 8.75M |

Postcode keys: +0.1 point overall, no change for India (each finds ~4.8% of true pairs alone, mostly
pairs the other retrievers already find). The postcode extraction in the normalize stub is weak; AM's
module may change this.

Speed (calibration: 50,000 test records against the full test S1 index, per-record stages scaled):
Pass 2 ~122 min for the full test set; rare-word df 0.002 with name prefix 4 ~53 min; with prefix 3
~38 min (measured full run: 22-34 min). The name TF-IDF is now the slowest stage (~64% of per-record
time); the rare-word address retriever is ~8x cheaper than the Pass 2 trigram retriever.
