# Leaderboard gap diagnosis (Pass 6)

Pass 4 scored **0.78191** on the public leaderboard; our eval said 0.9274 (0.9170 corrected). Pass 5 eval: 0.9345
(0.9302 corrected). Top teams score 0.99+. Tools: `python -m src.gap_diagnosis table | parity`.

## Summary

No bug and no train/test difference that our data can show explains the gap. The test code path reproduces the
eval score exactly, the test predictions look like the eval ones in every country (France included), and the
model's measured behaviour projects to about 0.93 on test. What remains is outside what we can measure locally
(the test ground truth or the scorer). A diagnostic submission without France separates the two leading
explanations (below).

## 1. Test vs eval vs train truth, per country (Pass 5 model, t = 0.79)

| | test France | test India | test US | eval India | train truth India | eval US | train truth US |
|---|---|---|---|---|---|---|---|
| S1 | 259,452 | 809,986 | 663,106 | 88,340 | 883,188 | 132,343 | 1,323,633 |
| S1 with empty list | 9.8% | 9.2% | 4.8% | 12.3% | 5.6% | 7.0% | 5.6% |
| matches per S1, mean | 3.20 | 3.17 | 3.49 | 2.84 | 3.47 | 3.16 | 3.46 |
| S1 with 1 / 2 / 3 / 4 / 5+ | .09/.16/.21/.20/.24 | .10/.17/.22/.20/.23 | .07/.16/.23/.23/.27 | .11/.20/.22/.18/.18 | .05/.17/.24/.22/.26 | .08/.20/.24/.20/.21 | .05/.17/.24/.22/.26 |
| candidates per record (pruned) | 5.10 | 4.44 | 4.86 | 4.03 | - | 5.37 | - |
| records assigned to an S1 | 57.8% | 54.3% | 60.6% | 69.6% (*) | - | 76.5% (*) | - |
| chosen pairs p >= 0.99 | 85.8% | 76.2% | 83.8% | 80.3% | - | 87.9% | - |
| exact name_norm among chosen | 34.9% | 16.8% | 25.1% | 18.1% | 15.8% (true pairs) | 27.4% | 26.0% (true pairs) |

(*) the eval side over-represents matched records (every val-matched record plus a sample of the rest), so its
assignment rate is not comparable.

Flags: **nothing sharp**. France looks like India and US; the per-S1 match distribution follows train truth in
every test country; a row misalignment between features and ids would push exact-name agreement toward 0, and it
sits where eval and truth are. The only difference is density: test has 5.8 S2/S3 records per S1 (train 4.7).

**Density does not explain it.** On the eval side at t = 0.79, matched records are assigned 86-88% of the time
(0.4% to a wrong S1) and never-matched records 4.3% of the time. Solving test's 5.71M assignments with these rates
gives about 6.35M matched test records (3.66 per S1, train: 3.46) and 3.62M never-matched ones, hence about 180k
false positives and a projected test F0.5 near 0.93.

## 2. Country strings

Each source has exactly `India`, `US`, `France` (test) and `India`, `US` (train): no case, whitespace, accent or
spelling variants, and normalization maps them to `india`, `us`, `france` one to one. Every S1 is blocked in its
own country group; 99.6% of test records (9,932,128 of 9,969,589) got candidates (records whose country has no S1 get none, and there are
no such countries).

## 3. End-to-end parity

The exact test code path (`iter_features` -> `predict` -> `decide` -> `write_grouped` from `src/run_pipeline.py`)
run on the eval half-B data wrote a real matching_results-format TSV; an independent scorer (csv module only,
reading that TSV and `train_ground_truth.tsv` from disk) gives **0.934454**, the sweep gives 0.934454 (difference
2e-13). No misalignment, id conversion, part-order or context bug in the test path.

## 4. France

25 random France S1 with their matches and 25 France records with their top 3 candidates
(`logs/pass6_france_samples.txt`, not committed): matches are overwhelmingly the same business (same street and
number, name equal up to accents, legal form or typos); a few are doubtful (same street, different number). AM's
normalize.py keeps accents (24.5% of France names and 18.2% of addresses carry one, often injected noise such as
"Çomite", "Ànciens"), does not recognize EURL or SCI as legal forms, does not canonicalize rue / r / bd / avenue,
and French addresses in the sample carry no postcode. This costs France some similarity but the model still
scores most France true-looking pairs at p ~ 1.0; it does not make France near 0.

## 5. Ids and format

S1 ids in matching_results.tsv equal test_source1.tsv exactly (1,732,544, unique). Ids go through int64 keys
(`src.data.id_key`), which is lossless here: every id is `S<1-3>-<digits>` with no leading zero (checked in
Pass 4), and the validator's `--check-ids` confirms every matched id exists in test S2/S3. 20 random predicted
pairs shown side by side are the same business in 17 or 18 of 20 cases. The Pass 4 upload (`output/submission_pass4.zip`)
contains the intact Pass 4 file (1,732,544 rows, byte size identical to the Pass 4 run).

## Causes, evidence, estimated impact, fix

| Cause | Evidence | Leaderboard impact | Fix |
|---|---|---|---|
| **Unexplained, external to our data** (test truth or scorer differs from train truth) | Sections 1-5: everything we can measure predicts ~0.93 | the whole 0.15 gap | Diagnostic submission below |
| Candidate: France scores near 0 on the leaderboard | 0.85 x 0.92 = 0.78 matches exactly; France is 15.0% of test S1 | up to 0.14 | depends on why; nothing local points to it |
| Blocking recall ceiling | 10% of true pairs never become candidates (val); leaders at 0.99 must recover them | ~0.03-0.05 on its own, not the 0.15 gap | Pass 7 (docs/am_pipeline_review.md) |
| French normalization (accents, EURL/SCI, rue/r/bd) | Section 4 | small | Pass 7 normalize work |
| **Bug found and fixed:** `core.autocrlf` rewrote `models/lgbm.txt` to CRLF on checkout, which LightGBM cannot load | parse error on load after the Pass 5 merge | none on submitted files (both were produced before the rewrite); breaks the model for anyone checking out main on Windows | `.gitattributes`: `models/*.txt -text` |

## Diagnostic submission

`output/diag_no_france_matching_results.tsv`: the Pass 5 file with every France S1 row emptied (validator PASS).
If France currently scores ~0, this should score about **0.79** (France singletons turn into 1.0, nothing is
lost); if France scores like the rest (~0.92), it should drop to about **0.65**. Either answer localizes the gap.
The Pass 5 file itself (`output/matching_results.tsv`) should score slightly above Pass 4 (eval +0.013).
