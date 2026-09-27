# Pass 10: loss breakdown, structure checks, sibling rescue

Time-boxed exploration, 17:00-17:25. Everything was read-only on saved outputs: `dist/`, `output/`, the Pass 7
predictions and the chainfix outputs are unchanged. Code: `src/pass10.py` (`loss | structure | siblings`).
The eval decision is chainfix at t = 0.80, and the chainfix eval delta is now saved to
`data/cache/pred_eval_chainfix_delta.parquet`. Peak memory was 3.98 GB for the chainfix eval rerun, and 3.4-3.7 GB
for every Pass 10 step.

## 1. Where the eval F0.5 is lost (chainfix, 220,683 half-B S1)

Corrected F0.5 is 0.95606, so 0.0439 is lost from 1.0. Buckets are per S1 and add up exactly. For S1 with at
least one TP, the loss is split into recall (scored with no false positives) and the extra cost of the false
positives.

| bucket | S1 | points lost (raw) | points lost (corrected) |
|---|---|---|---|
| S1 scored 0: **no true record was a candidate** | 3,273 | 0.0148 | 0.0148 |
| S1 scored 0: a true record's best S1 is this one but p < 0.80 | 438 | 0.0020 | 0.0020 |
| S1 scored 0: all true candidates went to another S1 | 156 | 0.0007 | 0.0007 |
| **partial recall** (TP > 0, some true records missed) | 41,838 | 0.0211 | 0.0211 |
| false positives on matched S1 (TP > 0) | 1,780 | 0.0017 | 0.0049 |
| singletons scored 0 (a false positive where nothing matches) | 91 | 0.0004 | 0.0004 |
| **total** | 47,249 | 0.0407 | 0.0439 |

The partial-recall S1 miss 57,136 true pairs:
- 43,276 (76%) were never candidates;
- 10,155 (18%) had this S1 as their best but p < t;
- 3,705 (6%) went to another S1.

**Blocking accounts for about 0.031 of the 0.044** (0.0148 + 0.76 x 0.0211). The threshold and model account for
about 0.006, and false positives for 0.005 (corrected). A 0.99 leaderboard implies near-complete candidate recall,
which the current channels do not reach (0.93 pair recall).

## 2. Structure checks on train (500,000 sampled true pairs vs the same pairs with shuffled records)

| signal | true pairs | shuffled | verdict |
|---|---|---|---|
| (a) mean relative row-position gap | 0.3334 | 0.3337 | none (uniform expectation 0.333) |
| (a) share of gaps < 0.001 / < 0.01 | 0.0021 / 0.0198 | 0.0020 / 0.0198 | none |
| (a) Spearman of row positions | 0.0013 | -0.0004 | none |
| (b) Spearman of id numbers | 0.0034 | 0.0001 | none |
| (b) same last 3 / last 2 digits | 0.0009 / 0.0097 | 0.0010 / 0.0100 | none |
| (b) relative id difference < 1% | 0.0098 | 0.0097 | none |
| (b) record id > S1 id | 0.4997 | 0.4994 | none |

In every source, ids are uniform over 1-999,999,999 (S1 217-999,998,822; S2 576-999,999,727; S3 10-999,999,866).
**Neither file order nor ids carry the match, so there is no artifact to exploit or disclose.**

**(c) Siblings of missed records** (65,928 true records of B S1 not correctly assigned, among 1.64M eval records):

| sibling key | FN records with a sibling correctly assigned to the true S1 | same, random S1 baseline |
|---|---|---|
| identical addr_core + first house number | 7.2% | 0.0% |
| identical name_core | 23.1% | 0.0015% |

**(d) Within-source duplicates.** 611,750 true records share their S1 and source with another true record, and
13.4% of them are identical to it in both name_core and addr_core. The model treats such identical groups
consistently: 96.1% are all assigned, 2.9% none and 1.0% split.

## 3. Sibling-propagation rescue (tuned on eval, not adopted)

**Rule.** An unassigned record joins S1 X when a sibling (same key) is assigned to X with p >= p_sib, all such
siblings agree on X, and a second similarity is at least s. Adoption bar: +0.002 corrected.

| key | best setting | records added | correct (among those to B S1) | corrected F0.5 gain |
|---|---|---|---|---|
| addr_core + house number, name token_set >= s | p_sib 0.95, s 0.5 | 3,828 | 3,011 of 3,194 (94%) | **+0.0005** |
| addr_core + house number | p_sib 0.95, s 1.0 | 1,027 | 825 of 853 (97%) | +0.0002 |
| name_core, address token_set >= s | p_sib 0.95, s 0.9 | 3,514 | 1,245 of 2,013 (62%) | -0.0019 |
| name_core, no address check | p_sib 0.95, s 0.0 | 24,696 | 4,226 of 12,936 (33%) | -0.0186 |

- **Address key:** precise (94-97%), but it covers only the 7.2% of missed records that have such a sibling. Even
  a perfect rescue of them would be worth about +0.0008.
- **Name key:** covers more, but chain names put identical names on many S1, so precision collapses and the x10
  false-positive correction punishes it.
- **Result:** no setting reaches +0.002. **Nothing was adopted**, no `chainfix_sib` variant was written, and
  `src/package.py` is unchanged.

## Takeaways

- **The missing points are candidate recall, not decision rules or hidden data structure.** The next lever is
  blocking: extend the chain-name join to crowded names that already have candidates, add an uncapped
  name + street channel, and add transliteration or phonetic keys for India. Each needs a re-block, which was out of
  scope in this pass.
- **The name-core sibling signal (23% of misses) is real but not actionable as a post-rule.** It would need to
  enter the model as a feature, for example "a same-name record is assigned here with high p", which means a
  re-score and a retrain.
