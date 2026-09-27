# France diagnosis (Pass 8b)

Leaderboard 0.8917 (Pass 7) against eval 0.9531 corrected. France is 15.0% of test S1 (259,452 of 1,732,544) and
absent from train, so eval says nothing about it. Everything here comes from the saved Pass 7 outputs and
predictions (`python -m src.france_diag table | nofrance | samples | checks`, peak <= 3.4 GB); nothing was
re-blocked or re-scored in bulk.

## Summary

- **The France model output looks healthy.** Its chosen pairs are the most confident of any country (82% at
  p >= 0.999) and have the highest exact name_core agreement (63%). Its match-count distribution matches India, US and train truth.
- **France's defect is in blocking, not in the model.** 10,610 France S1 (4.1%, against 1.9% India and 0.1% US)
  get **no candidate at all**. 99.3% of them carry a chain name shared by more than 50 S1 (median 104: *Bordeaux Club*
  x307, *Nantes Club* x285, *Lille Club* x232). 96.4% have a test record with the identical name_core and
  house number, and in a sample of 400 the street matches too for 94.8%. These are obvious duplicates that blocking
  crowds out: exact-key blocks are capped at 50 S1, and the TF-IDF name channel keeps the top 5 S1 per record,
  which ties among identical names. Each of these S1 scores 0 today.
- **The model is not missing obvious matches among the candidates it has.** Empty France S1 with candidates, and
  pairs just below t, are overwhelmingly *sibling* businesses: the same street at a different house number with a
  suffix word (*Holding*, *International*, *Développement*, *Groupe*, *Participations*), or name-only records whose
  chain name fits dozens of S1. On the eval side the same patterns in the 0.3-0.8 band are true matches only 26-47%
  of the time (other records weighted x10). Rejecting them at t = 0.8 is right for F0.5.
- **France alone can explain the gap only if France scores about 0.54** while the rest scores its eval value
  (0.953). The blocking defect above is worth about 0.04 of France's score (4.1% of its S1), not 0.4. The
  diagnostic submission decides it (formula below).

## 1. Per-country profile, Pass 7 (t = 0.80)

| | test France | test India | test US | eval India | eval US |
|---|---|---|---|---|---|
| S1 | 259,452 | 809,986 | 663,106 | 88,340 | 132,343 |
| S1 with no candidate (count) | **10,610** | 15,276 | 670 | 4,764 | 2,846 |
| S1 with no candidate | **4.09%** | 1.89% | 0.10% | 5.39% (*) | 2.15% (*) |
| S1 with an empty list | 7.98% | 7.85% | 4.70% | 10.04% | 6.23% |
| ... of which have candidates | 3.89% | 5.96% | 4.60% | 4.65% | 4.08% |
| matches per S1, mean | 3.39 | 3.22 | 3.50 | 3.00 | 3.27 |
| S1 with 0 / 1 / 2 / 3 / 4 / 5+ | .080/.070/.152/.220/.218/.260 | .079/.091/.174/.227/.204/.226 | .047/.063/.162/.239/.226/.263 | .100/.096/.194/.228/.188/.194 | .062/.072/.189/.246/.206/.225 |
| chosen p 0.8-0.9 / 0.9-0.95 / 0.95-0.99 | 3.1 / 2.5 / 5.3% | 3.2 / 3.0 / 7.8% | 3.0 / 2.5 / 6.4% | 1.2 / 1.4 / 5.2% | 1.0 / 1.2 / 4.3% |
| chosen p 0.99-0.999 / >= 0.999 | 7.0 / **82.1%** | 13.1 / 73.0% | 10.9 / 77.3% | 11.3 / 80.8% | 9.2 / 84.3% |
| S2/S3 records assigned | 61.3% | 55.2% | 60.8% | 72.8% (*) | 72.5% (*) |
| record-best pairs with 0.3 <= p < t, per S1 | 0.36 | 0.31 | 0.42 | 0.05 (*) | 0.07 (*) |
| exact name_core among chosen pairs | **63.2%** | 43.4% | 51.9% | 44.8% | 53.0% |
| eval F0.5 corrected | - | - | - | 0.9209 | 0.9745 |

(*) Not comparable. The eval side queries every record of the half-B S1 but only 10% of the other records,
so eval has fewer candidates per S1 and more matched records per query.

The 26,556 S1 without any candidate split into 10,610 France, 15,276 India and 670 US.

One more observation, which concerns the gap outside France. On test, India and US accept 2.5-3x more pairs in
0.8-0.9 than eval does (3.0-3.2% of chosen pairs against 1.0-1.2%). The full-density test candidate lists hold
more borderline pairs than the eval side, and the x10 correction models their false positives only on average.

## 2. Diagnostic submission

`output/diag_p7_no_france.tsv` (zipped as `output/diag_p7_no_france.zip`, 33.1 MB, one member
`matching_results.tsv`) is the Pass 7 file with all 259,452 France rows emptied (238,758 had matches). The
organizer validator passes it with and without `--check-ids`: 1,732,544 rows, 354,202 empty.

## 3. Backing out France's own score

Let L = 0.8917 be the Pass 7 leaderboard score and D the diagnostic's score. Let f = 0.150 be the France share of
the public-leaderboard S1, s the France singleton rate, F France's mean F0.5 in Pass 7, and R the rest's.

    L = (1 - f) R + f F
    D = (1 - f) R + f s              (an emptied France row scores 1 on a singleton, 0 otherwise)

    F = s + (L - D) / f
    R = (D - f s) / (1 - f)

With s = 0.0558: **F = 0.0558 + (0.8917 - D) / 0.15** and **R = (D - 0.00837) / 0.85**.

| D (diagnostic) | F (France) | R (rest) | reading |
|---|---|---|---|
| 0.7663 | 0.892 | 0.892 | France scores like the rest: the gap is not France |
| 0.7900 | 0.734 | 0.920 | France part of the gap |
| **0.8185** | **0.544** | **0.953** | the rest matches eval: France explains the whole gap |
| 0.9001 | 0.000 | 1.049 | impossible (R > 1), so F >= 0.28 |

Sensitivity: dF/dD = -1/f = -6.7, so leaderboard rounding (+-0.0001) moves F by +-0.0007. If the public France
share is 14% or 16% instead of 15%, F moves by about +-0.03 near F = 0.55. F moves one for one with s.

The singleton rate: train truth has 5.58% singletons, with India and US both at 5.6%, which points to a fixed
generator rate. France should share it. A model-based estimate, prod(1 - p) per S1 with p recalibrated for the
0.5 negative subsampling, gives 7.3% France, but it also gives 7.1% India and 3.9% US on test, where the truth is
about 5.6% for both. Its error is larger than any France-specific signal, so **use s = 0.0558**.

## 4. Samples (logs/pass8b_france_samples.txt, not committed)

The 60 samples' features were rebuilt through the test code path, and re-predicting them gives the saved Pass 7
probabilities exactly (max difference 0). The listed reasons are the four most negative LightGBM contributions.

**A. 30 France S1 with an empty list that do have candidates.**

| Pattern | Count | Example | Verdict |
|---|---|---|---|
| Sibling at a neighbouring number: name + suffix word or legal-form swap, same street, number off by 1-25 | ~17 / 30 | *Devers (France) Club, 5 Quai Hubert Prom* vs *Devers (France) Club International, 8 Q. Hubert Prom* | ambiguous |
| Name-only record (empty address) whose chain name fits 20-50 S1 | ~9 / 30 | *Communale Club SCI, 62 Av de Candau* vs *COMMUNALE CLUB SCI, (no address)*, 50 candidates | cannot be placed |
| Nothing plausible | ~4 / 30 | | likely true singletons |

On the siblings, the model's main reasons are num_jaccard = 0 (-2.3 to -3.2 logits), then name_token_sort and
house_number_conflict. On the name-only records they are addr_token_set = -1 and the record's many candidates
(rec_n_cands, rec_rank_tfidf_nameaddr_cos).

The sibling verdict is ambiguous because house numbers carry generated noise: 94-96% of accepted conflict pairs
are true.

**B. 30 France record-best pairs with 0.3 <= p < 0.8.** 28 of 30 carry a house-number conflict, and 25 are
the sibling pattern (near-identical name, same street, different number), e.g. *Union du Solidarite, 612 c Grande Rue du Petit Courgain* vs
*Union du Solidarite SARL, 616 C Grande Rue du Patit Courgain*. The rest are name-only records or different names at the same address.

**Eval side, record-best pairs (`checks`).** This shows how often these patterns are true matches where labels
exist:

| band | pattern | India: true (x10 corrected) | US: true (x10 corrected) |
|---|---|---|---|
| 0.3 <= p < 0.8 | house-number conflict | 0.42 (0.33) | 0.36 (0.26) |
| 0.3 <= p < 0.8 | record has no address | 0.54 (0.52) | 0.50 (0.47) |
| 0.3 <= p < 0.8 | other | 0.37 (0.29) | 0.40 (0.31) |
| p >= 0.8 | house-number conflict | 0.98 (0.96) | 0.97 (0.94) |

The model is calibrated on exactly these patterns in India and US. A lower France threshold would add pairs that
are true about a third of the time, which lowers F0.5.

Accents are not the problem either. `normalize.py` (Pass 7) folds them (*Établissements* -> *etablissements*,
*Pàul* -> *paul*), and legal forms such as SARL, SAS, EURL, SCI and SASU are stripped from name_core.

Two France-only normalization gaps remain, both with zero cases in India and US:
- **Dotted legal forms survive.** *S.A.S*, *S.A.S.* and *S.A.R.L* become `s a s` / `s a r l` in name_core, e.g.
  *Bordeaux Club S.A.S* -> `bordeaux club s a s`. This affects 2.85% of France S2/S3 records and costs them name similarity.
- **`N°24` becomes `ndeg24`** in addr_norm (1.65% of France records). This is harmless to the number features,
  because the `numbers` column still holds 24. Only address-token overlap loses one token.

## 5. Fix

**Most promising: a supplemental exact channel for crowded chain names.** It is not a full re-block. For every S1
whose name_core is shared by more than 50 S1 in its country, add as candidates the records with the identical
(name_core, first house number). The join takes seconds, and the pairs are appended to the pruned candidate set
and `candidate_pairs.tsv`.

Candidate lists change only for the records involved, but rec_rank_* / rec_gap_* / rec_n_cands depend on a
record's whole list. So the parts holding those records must be re-featurized and re-scored:
- France: all 8 parts, about 8.5 min.
- India: 15,276 S1 without candidates, 55% of which have a name + number record. Optional, about 28 min.

Expected gain, if the fixed S1 then score about 0.9:
- France: about +0.0055 on the leaderboard (10,610 x 0.9 / 1.73M).
- India: about +0.004.

The same crowding also caps recall for chain-name S1 that did get some candidates (4% of France S1 with
candidates carry a name shared by more than 50), so the gain could be somewhat larger.

**Cheap but small: strip dotted legal forms** (`s a s`, `s a s u`, `s a r l`, `e u r l`, `s c i`, `s n c`, `s a`
at the end of name_core). It changes France records only (2.85%), so training features and the model are
unchanged. It needs `norm_cache test` (2.7 min) plus re-featurizing and re-scoring the 8 France parts.

Take care with the second step. `run_pipeline` wipes every cached part prediction when `normalize.py`'s hash
changes. Score the France parts into a separate directory, or re-tag the cache on purpose, so India and US are
not re-scored (about 115 min).

**Not recommended:** a France-specific threshold. The near-miss band is mostly sibling hard negatives, as the eval
rates show.

If the diagnostic returns D near 0.766, the rest of the gap is outside France. The first place to look is the
2.5-3x excess of borderline accepts in test India and US (section 1).
