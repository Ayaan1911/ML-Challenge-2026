# Pass 11: additive blocking channels on top of chainfix (measured, not adopted)

Pass 10 showed that blocking accounts for about 0.031 of the 0.044 eval loss, so this pass tried two more additive
candidate channels, built the same way as chainfix (`src/chainfix2.py`). There was no re-blocking and no
retraining: the Pass 7 model and the t = 0.80 decision were kept. Blocking columns came from the blocking code, only
the records involved were re-scored, and control records were checked. Nothing in `dist/`, `output/` or the cached
predictions changed.

## Channels (within country, on top of the Pass 7 + chainfix candidates)

- **a, extended chain join.** Every S1 whose space-free name_core key exceeds the blocking cap (50 S1) gets the
  records with identical name_core and identical first house number, uncapped.
- **b, name + street.** Identical name_core plus an identical street signature (addr_core words without digits,
  sorted and deduplicated). Keys held by more than 200 S1 are skipped.
- **c, India consonant skeleton + postcode.** Not built; see the decision below.

Unlike chainfix, these pairs can point at S1 that already have candidates, which changes those S1's context
features (`s1_n_cands`, `s1_rank`). Every record with a row at such an S1 was therefore re-scored as well
(`chainfix.rescore(s1_context=True)`), not only the records that gained pairs. That came to 33,899 records in
India and 8,031 in the US (a + b). Controls exclude those records. 3,138 and 3,016 control rows reproduced their
saved probabilities exactly.

## Eval result (220,683 half-B S1, 763,643 true pairs)

New pairs by channel. Pairs pointing at other train S1 are irrelevant to scoring but take part in each record's
best-S1 choice.

| country | channel | pairs added | pairs to B S1 | true | true rate on B |
|---|---|---|---|---|---|
| India | chainfix (base) | 1,688 | 1,688 | 1,226 | 73% |
| India | a | 22,837 | 2,335 | 688 | 29% |
| India | b | 143 | 83 | 82 | 99% |
| US | chainfix (base) | 105 | 105 | 97 | 92% |
| US | a | 1,869 | 236 | 68 | 29% |
| US | b | 1,036 | 430 | 400 | 93% |

Scores at t = 0.80:

| | candidate pair recall | F0.5 corrected | gain vs chainfix | India corrected | US corrected | F0.5 raw |
|---|---|---|---|---|---|---|
| Pass 7 | 0.9300 | 0.95306 | - | 0.92093 | 0.97452 | 0.95628 |
| chainfix | 0.9317 | 0.95606 | - | 0.92792 | 0.97484 | 0.95928 |
| chainfix + a | 0.9327 | 0.95659 | +0.00053 | 0.92913 | 0.97492 | 0.95980 |
| chainfix + a + b | 0.9333 | **0.95704** | **+0.00098** | 0.92948 | 0.97544 | 0.96026 |

**Decision: not adopted.** The best combination gains +0.0010, half the +0.002 bar, so no `chainfix2` test variant
was built and `src/package.py` is unchanged. Channel c was not built: a and b together recover only 1,238 true
B pairs (0.16 recall points), and a skeleton key capped at 50 would face the same crowding that limits channel a.

## Why the gain is small

- **Channel a is imprecise.** Crowded chain names share house numbers across streets, so only 29% of its B pairs
  are true. The model rejects most of the false ones, but they also compete with true candidates in each record's
  best-S1 choice.
- **Channel b is precise but rare.** 93-99% of its B pairs are true, but an identical street signature across
  sources is uncommon, because city, region and ordering differ between S1 and S2/S3 addresses.
- **Candidate recall barely moves:** 0.9317 -> 0.9333. The remaining 6.7% of true pairs are not reachable with
  exact joins on normalized fields. They need fuzzy retrieval (for example a larger name top-k, or address
  retrieval that is not rare-word-only), which means a re-block.

## Code changes kept

- `src/chainfix2.py`:
  - `eval <country>`: one process per country (memory).
  - `eval-score`: control checks and corrected F0.5 per combination.
  - `test <country> <combo>`: ready but not run.
- `src/chainfix.py`: `rescore(..., s1_context=True)` re-scores records affected through S1 context. The default is
  off, so chainfix is unchanged.
- `src/variants.py`: `write_variant` accepts a chain of deltas and reports changes against the previous variant
  (`python -m src.variants chainfix2 ...`).

**Runtime and memory.**
- India eval: 2.6 min, 3.8 GB. US eval: 2.8 min, 3.8 GB. Scoring: 0.6 min, 3.4 GB.
- The first attempts were stopped by the memory guard while the browsers held about 5.5 GB. A first scoring run
  stalled on `np.isin` over 8M object-dtype ids; it was stopped and replaced by a hashed pandas `isin`.
