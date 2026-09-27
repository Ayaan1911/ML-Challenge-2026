# Pass 8c: threshold variants and chain-name candidate fix

Pass 7 (leaderboard 0.8917) stays the submission in `output/`. Every variant below lives in its own folder under
`output/variants/` and was built from the saved Pass 7 predictions. The chain-name fix adds a delta scored into
`data/cache/pred_test_chainfix/`; `data/cache/pred_test/` was not written. Each variant passes the organizer
validator with and without `--check-ids`, and `src.check_candidates` with and without `--check-ids`. Each zip holds
`matching_results.tsv` (the leaderboard upload).

## Variants

| variant | zip | t | eval F0.5 corrected (Pass 7: 0.9531) | matched pairs | France +/- | India +/- | US +/- |
|---|---|---|---|---|---|---|---|
| t085 | `output/variants/t085/submission_t085.zip` (37.2 MB) | 0.85 | 0.9528 (-0.0003) | 5,721,953 | +0 / -12,679 | +0 / -35,687 | +0 / -32,415 |
| t090 | `output/variants/t090/submission_t090.zip` (36.7 MB) | 0.90 | 0.9518 (-0.0013) | 5,622,711 | +0 / -27,625 | +0 / -82,722 | +0 / -69,676 |
| **chainfix** | `output/variants/chainfix/submission_chainfix.zip` (37.8 MB) | 0.80 | **0.9561 (+0.0030)** | 5,834,852 | +25,576 / -752 | +7,426 / -132 | 0 / 0 |
| chainfix_t085 | `output/variants/chainfix_t085/submission_chainfix_t085.zip` (37.4 MB) | 0.85 | 0.9558 (+0.0027) | 5,753,904 | +25,348 / -13,349 | +7,369 / -35,783 | +0 / -32,415 |

Pass 7 has 5,802,734 matched pairs: France 878,932, India 2,604,346, US 2,319,456. A pair is counted as added or
removed when a record's S1 link appears, disappears or moves; a move counts once in each column. The chainfix
folders also hold their own `candidate_pairs.tsv`: Pass 7's 46,794,809 pairs plus the 125,819 new ones.

## Eval evidence (eval side, 220,683 half-B S1)

| | t | F0.5 corrected | F0.5 | India corrected / raw | US corrected / raw | precision corrected | recall | singleton acc |
|---|---|---|---|---|---|---|---|---|
| Pass 7 | 0.80 | 0.9531 | 0.9563 | 0.9209 / 0.9244 | 0.9745 / 0.9776 | 0.9776 | 0.9109 | 0.9929 |
| Pass 7 | 0.85 | 0.9528 | 0.9554 | 0.9204 / 0.9233 | 0.9744 / 0.9768 | 0.9821 | 0.9075 | 0.9944 |
| Pass 7 | 0.90 | 0.9518 | 0.9536 | 0.9192 / 0.9214 | 0.9735 / 0.9752 | 0.9868 | 0.9021 | 0.9957 |
| chainfix | 0.80 | **0.9561** | 0.9593 | 0.9279 / 0.9314 | 0.9748 / 0.9779 | 0.9776 | 0.9127 | 0.9927 |
| chainfix | 0.85 | 0.9558 | 0.9584 | 0.9274 / 0.9303 | 0.9748 / 0.9771 | 0.9821 | 0.9092 | 0.9942 |
| chainfix | 0.90 | 0.9548 | 0.9566 | 0.9262 / 0.9284 | 0.9738 / 0.9755 | 0.9868 | 0.9038 | 0.9956 |

The threshold variants rest on a test-only signal. Test India and US accept 2.5-3x more pairs at p 0.8-0.9 than
eval does (docs/france_diagnosis.md), but eval itself prefers t = 0.80. Only the leaderboard can say whether
test's extra borderline pairs are mostly false.

## Chain-name candidate fix (src/chainfix.py)

**Rule.** It is the same on eval and test. For every scored S1 with no candidate, add as candidates the records of
its country that already have candidates and share its exact name_core and first house number, with no cap.

**Blocking columns for new pairs.** They come from the blocking code itself:
- key flags from the same `KeyIndex` caps;
- true TF-IDF cosines from the same per-country S1 indexes;
- `addr_word_idf` = 0, since the rare-word retriever did not return these pairs.

**Rescoring.** Every row of each affected record (old plus new) is re-featurized and re-scored. Record context
depends only on the record's own rows, and the target S1 had no rows before, so no other row's features change.
The Pass 7 decision is then applied unchanged.

**Self-checks.** Both passed on every run, and a run aborts on any difference:
- The blocking columns of 2,000 stored pairs per country are recomputed with zero difference.
- Re-featurized control records reproduce their saved probabilities exactly: 2,751 rows on eval, 1,428 in France
  and 1,280 in India.

| | S1 without candidates | ... that gain a pair | new pairs | new pairs at p >= 0.8 / true | records re-scored |
|---|---|---|---|---|---|
| eval India | 4,764 | 1,084 | 1,688 | 1,226 true (73%) | 1,643 |
| eval US | 2,846 | 52 | 105 | 97 true | 105 |
| test France | 10,610 | 10,256 | 102,538 | 27,807 at p >= 0.8 | 35,112 |
| test India | 15,276 | 8,733 | 23,281 | 7,432 at p >= 0.8 | 19,223 |

US was not run on test: it has only 670 S1 without candidates, and eval US gains +0.0003.

On eval, India's corrected F0.5 rises by 0.0070 and US's by 0.0003. France cannot be measured, but it carries the
largest effect: 10,256 of its 10,610 candidate-less S1 gain pairs, and the France gain is +25,576 matched pairs.
If those S1 are not singletons and now score about 0.9, the leaderboard gain is about +0.0055 from France and
+0.003 from India.

## Runtime and memory

| step | time | peak |
|---|---|---|
| threshold variants t085 / t090 (write, 4 validations, zip) | 5.5 + 5.5 min | 3.2 GB |
| chainfix eval (India + US, both checks, before/after scoring) | 8.8 min | 4.3 GB |
| chainfix test France / India | 2.6 + 6.9 min | 2.0 / 3.7 GB |
| chainfix / chainfix_t085 variants (incl. candidate file) | 6.5 + 6.4 min | 4.1 GB |
| **total compute** | **about 42 min** | |

The wall clock was about 1 h (14:55-15:55). That includes two failed eval attempts:
- **Memory guard stop.** Six joblib workers hashing S1 texts held about 2 GB beyond the 3.3 GB process. The guard
  checks free system memory, so it stopped the run, and the workers were left orphaned. `block_cols` now hashes
  in-process.
- **Control-check bug.** Controls were drawn from both sides of the val split, but saved predictions exist for the
  eval side only.

The eval run and the chainfix variant writes peaked at 4.1-4.3 GB, slightly above the 4 GB target.

## Upload order

1. `chainfix`: the only variant with positive eval evidence (+0.0030 corrected), and it targets the France defect.
2. `chainfix_t085`, if chainfix improves: it tests the stricter threshold on top.
3. `t085` / `t090`: threshold only, against eval's preference.
