# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** neural forge  
**Team Members:** Aditya Malik (team leader; Apex Institute of Technology, Chandigarh University), Aryan Rajan (K L University, Guntur), Mohammad Ayaan (M S Ramaiah Institute of Technology, Bengaluru)  
**Submission Date:** 2026-09-27  
**Submitted variant:** `chainfix`. Eval F0.5, corrected: 0.9561. Public leaderboard F0.5: 0.897.

---

## 1. Executive Summary

We link each Source 1 business record to the Source 2 / Source 3 records of the same business. The pipeline has
four stages:
- **Normalization:** a rule-based, multilingual, noise-hardened step (anyascii transliteration, legal-form and
  street-word tables, postcode, house-number and landmark extraction).
- **Blocking:** eight exact-key channels and two TF-IDF retrievers, made fast by prefix filtering and pruned by
  score. It generates 46.8M test candidates (about 4.7 per record) and keeps 93% of true pairs.
- **Scoring:** a LightGBM classifier over 60 features.
- **Decision:** each record is assigned to its single best S1 if p >= 0.80.

The main contribution is **record-relative context features**: how an S1 ranks against the record's other
candidates. They carry 82% of the model's gain and turn a pairwise classifier into a competitive assignment.

The submitted chain-name variant scores 0.9561 corrected macro F0.5 on eval and 0.897 on the public leaderboard
(Pass 7: 0.9531 and 0.8917). Everything runs on a 16 GB laptop without a GPU, external data or APIs.

---

## 2. Methodology

### 2.1 Problem Analysis

**Data.** Test has 1,732,544 S1 records (India 46.8%, US 38.3%, France 15.0%) and 9,969,589 S2/S3 records.
Train has 2,206,821 S1, only India (40%) and US (60%), with 10.3M S2/S3. EDA findings that shaped the design:
- **Hub structure.** The ground truth maps each S1 to a set of S2/S3 ids, 3.46 per S1 on average in train. 5.58% of
  train S1 are singletons (empty set, score 1 only if we predict nothing).
- **One S1 per record.** No S2/S3 record belongs to more than one S1. The task is therefore an assignment: each
  record picks at most one S1. We enforce this in the decision layer.
- **Country consistency.** A true pair always shares the country string, exactly `India`, `US` or `France`, with
  no variants. All processing is per country.
- **S1 is clean, S2/S3 are noisy.** 2.7-3.4% of S2/S3 addresses are empty. S2/S3 carry case changes, abbreviations,
  reordered address parts, typos, bracketed text, URLs as names, trailing phone numbers, junk tokens (N/A, <NULL>,
  ##), doubled words and Indic scripts. Transliterating those scripts takes Hindi names to 18-28% of India names.
- **France has no training data.** A noise audit (train vs test, per source and country) finds India and US test
  noise within a few percent of train. France differs: 16-28% of France names and addresses carry accented Latin
  letters, real ones plus injected ones such as "Çomite" or "Ànciens", which train never shows in addresses.
  Several French specifics also appear: legal forms (SARL, SAS, SASU, EURL, SCI, SNC), street words (rue/r, av,
  bd, imp, all), no postcodes in most addresses, and "N°" numbering.
- **Density.** Test has 5.8 S2/S3 records per S1 against train's 4.7, so candidate lists are more crowded on test.

**Metric.** Macro F0.5 over all S1: per S1, F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP), and 1 when both truth and
prediction are empty. Precision weighs 4x more than recall, and every S1 counts equally, singletons included.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + assignment-style decision layer (hybrid rule/ML).  
**Core Innovation:** Context features relative to a record's own candidate list (rank and gap to the best
candidate on each similarity), plus S1-side context. With them, a gradient-boosted pair classifier decides "which
of these S1 is this record" instead of "are these two similar". A second contribution is engineering: a
prefix-filtered sparse top-k retriever and a streamed, resumable, memory-guarded pipeline let a 16 GB laptop
process 12M records and 142M candidate pairs.

Pipeline (each stage a `python -m src.<module>` command; see README.md):

```
TSV -> parquet cache -> normalize (NormRecord) -> blocking per country (keys + TF-IDF + rare words)
    -> prune (keep key hits + top 3 by score per record) -> 60 features -> LightGBM p
    -> decision: best S1 per record, p >= t -> matching_results.tsv / candidate_pairs.tsv
```

### 2.3 Normalization (`src/normalize.py`)

Each raw record becomes a NormRecord with these fields:
- `country_norm`
- `name_norm`, `name_core` (the name without its legal form), `legal_suffix`, `dba_name`
- `addr_norm`, `addr_core` (the address without postcode and landmark phrases)
- `postcode`, `numbers` (house and unit numbers), `landmark`

The rule families:
- **Folding and transliteration.** Every non-ASCII character is transliterated with anyascii (ISC). This covers
  accents on any Latin letter, and Devanagari, Bengali, Gujarati, Gurmukhi, Tamil, Telugu, Kannada and Malayalam.
  Everything is then lowercased, and punctuation becomes spaces, so '##19821' becomes '19821'.
- **Noise removal in names.** URL protocols, `www.`, TLDs and a leading @ or # are stripped. So are trailing 6-12
  digit numbers (phone or registration numbers), junk tokens (N/A, <NULL>, ##, --, empty brackets) and
  consecutive doubled words. Bracket characters are removed but their content is kept, since it is usually part
  of the name.
- **Legal forms.** Tables for US, India and France: inc/corp/co/llc/llp/lp/plc/ltd/pvt/opc and
  sarl/sas/sasu/sa/eurl/snc/sci/scop. The table also maps transliterated Indic legal forms, counted on train
  S2/S3: praivet/praibhet/piraivet/praivrr -> pvt, limitet/limirrd/limtid -> ltd, elelpi -> llp, 'pra li' -> pvt
  ltd. A trailing legal form moves to `legal_suffix`; the rest is `name_core`.
- **DBA / trading names.** "d/b/a", "dba", "trading as", "t/a" and "aka" split the name into a primary name and
  a trade name (`dba_name`).
- **Addresses.** Street and unit words are canonicalized: street -> st, road -> rd, avenue -> av, boulevard -> bd,
  chemin -> ch, impasse -> imp, allee -> all, faubourg -> fbg. "r" -> "rue" only in French addresses. Indian city
  aliases are unified (bengaluru/blr -> bangalore, bombay -> mumbai, madras -> chennai, calcutta -> kolkata,
  gurugram -> gurgaon), and mahatma gandhi -> mg.
- **Postcodes.** Country-specific patterns: India 6 digits (with or without the space), US 5 (+4), France 5, and
  a generic 5-6 digit pattern for other strings. The postcode is removed from `addr_core`.
- **Numbers and landmarks.** The house and unit numbers other than the postcode go to `numbers`. Cue phrases
  ("near", "opp", "behind", "next to", "pres de", "face a") put the following phrase into `landmark` and out of
  `addr_core`.

All rules are per record, with no corpus statistics, so the normalization of a record never depends on which
other records are present. `src/norm_cache.py` applies them in parallel (six workers, 2.7 min for all 22M
records).

### 2.4 Blocking (`src/blocking.py`)

Per country, the S1 records form an index, and every S2/S3 record queries it with these channels:

| channel | key / method | cap |
|---|---|---|
| `key_name_nospace` | name_core with spaces removed | skip keys held by > 50 S1 |
| `key_sorted_tokens` | name_core tokens sorted | > 50 |
| `key_postcode_num` | postcode + first number | > 50 |
| `key_postcode` | postcode | > 50 |
| `key_word_postcode` | every address word (>= 3 chars or a number) x postcode | > 50 |
| `key_anchor` (Pass 7) | house number + first 3 name letters; postcode + first 3 letters | > 20 |
| `key_skip_bigram` (Pass 7) | name-token skip bigrams (brand words, stopwords removed) | > 20 |
| `key_prefix8` (Pass 7) | first 8 characters of the space-free core name | > 20 |
| name TF-IDF | char_wb 3-gram TF-IDF (sublinear tf) on name_core, top 5 S1 with cosine > 0.3 | - |
| rare-word address retriever | summed IDF of shared rare words (held by <= 0.2% of the country's S1) in name + address, top 5 | - |

**Prefix filtering.** A full sparse product visits every posting of every query trigram. Common trigrams touch
about 27k S1 per query, which would take hours at full scale. Each name query therefore probes the index with
only its 3 rarest trigrams (`sparse_dot_topn`), keeps the best 15 S1 by that partial score, and re-scores them
with the true full-vector cosine. The address side uses the rare-word posting lists, whose length is bounded by
construction.

**Merging and rescoring.** Pairs from all channels are merged. Every merged pair gets the true name cosine and the
true name + address cosine (`rescore_all`), so hits from the Pass 7 channels can compete in pruning.

**Pruning.** A record keeps every hit of the five original exact keys, plus its top 3 candidates by
`prune_score = name_cos + nameaddr_cos + addr_word_idf / 20`. The rare-word term keeps true pairs that only the
address retriever finds: 89.9% against 85.2% recall at n = 3. Pruning happens at read time, so n can be retuned
without re-blocking.

**Measured at real density.** Val queries run against the full 2.2M train S1 index, over 1,527,889 true val pairs:

| | pair recall | India | US | F0.5 ceiling | pairs per query record |
|---|---|---|---|---|---|
| Pass 3/5 blocking, pruned n = 3 | 0.8975 | - | - | 0.9476 | 4.8 |
| Pass 7, unpruned | 0.9361 | 0.8870 | 0.9690 | 0.9685 | 14.46 |
| **Pass 7, pruned n = 3 (used)** | **0.9298** | 0.8784 | 0.9642 | **0.9659** | 4.86 |

On test, 142,179,392 stored pairs are pruned to **46,794,809 candidate pairs**. That is the
`candidate_pairs.tsv` fed to the model. 1,705,988 of the 1,732,544 S1 have at least one candidate.

**Chain-name candidate step (Pass 8c, `src/chainfix.py`).** Generic chain names such as "bordeaux club" (307 S1)
or "nantes club" (285) exceed the exact-key caps and tie in the top-5 name retriever. As a result 10,610 France
S1 (4.1%) and 15,276 India S1 got no candidate at all, although for 96% of the France ones a record with the
identical name and house number exists (and in a sample, 95% also share the street). For every S1 without
candidates, the step adds the records of its country with identical `name_core` and first house number, with no
cap. Their blocking columns are computed with the blocking code itself, and only the records involved are
re-scored.

Self-checks, each of which aborts the run on any difference:
- Recomputed blocking columns of 2,000 stored pairs per country match the stored values exactly.
- Re-featurized control records reproduce their saved probabilities exactly.

Results:
- Test: 125,819 new candidate pairs (France 102,538, India 23,281); 10,256 France and 8,733 India S1 gain candidates.
- Eval: 73% of the new India pairs are true, and corrected F0.5 rises from 0.9531 to 0.9561.

### 2.5 Features (`src/features.py`, `src/matcher.py`)

There are 60 features per (S1, record) pair. Importance is the LightGBM gain share of the final model.

| group | # | gain share | features |
|---|---|---|---|
| Record-relative context | 13 | **0.823** | rec_n_cands; rank of this S1 among the record's candidates and gap to the record's best, for tfidf_name_cos, tfidf_nameaddr_cos, addr_word_idf, prune_score, name_token_set, addr_token_set |
| Name similarity | 19 | 0.064 | rapidfuzz ratio / token_sort / token_set on name_norm; Jaro-Winkler, token_set, partial, exact, word Jaccard and char-3 Jaccard on name_core; first word, acronym, legal-suffix match, best DBA match, length ratio, name-in-address; Pass 7: disjoint brand words, branch-modifier asymmetry (north/east/…), exact name with a missing address, space-free core ratio |
| Address and other | 13 | 0.067 | token_set / token_sort / partial / word Jaccard on addresses, IDF-weighted address words, postcode match and 3-digit prefix, number Jaccard, first-number match, landmark similarity, length ratio, country match, missing-field count; Pass 7: house-number conflict |
| Blocking | 12 | 0.033 | the eight key flags, name cosine, name + address cosine, rare-word IDF score, prune_score |
| S1-relative context | 2 | 0.014 | records pointing at this S1 and this record's rank among them by prune_score (weighted by sampling weight, so val matches the fully queried test) |
| Source | 1 | 0.000 | S3 vs S2 |

A feature that cannot be computed because a field is empty returns -1, never NaN. Features are symmetric in their
two records. Country is deliberately **not** a feature, since France is absent from training.

Top 10 features by gain:
1. rec_rank_tfidf_nameaddr_cos 0.587
2. rec_rank_prune_score 0.107
3. rec_gap_tfidf_nameaddr_cos 0.064
4. rec_gap_prune_score 0.035
5. num_jaccard 0.029
6. tfidf_nameaddr_cos 0.021
7. name_token_sort 0.017
8. rec_rank_addr_token_set 0.012
9. addr_token_set 0.010
10. house_number_conflict 0.009 (new in Pass 7)

### 2.6 Model

LightGBM binary classifier with these settings:
- 63 leaves, learning rate 0.05, min_data_in_leaf 100
- feature and bagging fraction 0.8, bagging frequency 1, L2 regularization 1.0, seed 42

Training data:
- The train side of the val candidate set (section 2.8).
- Negatives are subsampled at `NEG_RATE = 0.5`: 4,176,065 rows, 27.6% positive.
- Early stopping (100 rounds) on a holdout of 10% of train-side records, 800,375 rows, not subsampled.

The best iteration is **3,902 trees**, with holdout log loss 0.01553 (Pass 5: 2,998 trees, 0.02082). The model file
is 26.2 MB, and training takes 7.7 min with a 4.2 GB peak. Subsampling the negatives doubles the model's odds;
the threshold sweep absorbs this.

### 2.7 Decision layer (`src/postprocess.py`)

1. **One S1 per record.** Each record keeps only its highest-probability S1, following the EDA constraint.
2. **Threshold.** The pair is accepted if p >= t. t = 0.80 maximizes corrected F0.5 on the eval side, sweeping
   0.02-0.98 in steps of 0.01.
3. **Pass 8 rule study.** Three extra rules were tuned jointly with t on the saved eval predictions by coordinate
   descent over all 8 subsets (1,045 rule sets):
   - (b) a second chance for S1 left empty (take its best unmatched candidate if p >= t2);
   - (c) a relative margin (also keep pairs within m of the S1's best accepted p, above t3);
   - (d) a per-S1 cap k.

   The best subset (b + c) gained only +0.0007, below the adoption bar of +0.001, and cost 1.2 points of singleton
   accuracy. **No rule was adopted.** `models/decision.json` keeps t = 0.80 alone.

### 2.8 Validation design

The val candidate set is 20% of train S1 (441,365 S1), stratified by has-match x country, seed 42. It is built by
querying the full 2.2M train S1 index with every record matched to a val S1, plus a 20% sample of all other train
records. This makes blocking density realistic.

It is split **by query record**, never by pair:
- The val S1 are halved (A/B, seed 42). Records of A S1 form the **train side**; records of B S1 form the
  **eval side**.
- The sampled other records are split 50/50.
- Macro F0.5 is computed over the 220,683 half-B S1 (12,451 singletons). It includes true pairs lost by blocking
  or pruning.

**False-positive correction.** "Other" records on the eval side are a 10% sample (20% x 50%), so false positives
from them are under-counted 10x. The **corrected** F0.5 counts each such false positive as 10 and is the
selection criterion everywhere. It is still somewhat optimistic: it cannot add back the singletons that the
unqueried 90% would have hit.

**Parity check.** The exact test code path (features -> predict -> decide -> TSV writer), run on the eval data and
scored by an independent csv-only scorer, reproduces the sweep's F0.5 to 2e-13.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Exact keys: space-free name, sorted name tokens, postcode + house number, postcode, address word x postcode,
    house number + name prefix, postcode + name prefix, name skip-bigrams, 8-character name prefix.
  - Retrievers: char 3-gram TF-IDF on names (prefix-filtered top 5), and a rare-word IDF retriever on name +
    address (top 5).
  - The chain-name exact join (name + house number) for S1 without candidates.
  - All of it per country (section 2.4).
- **Candidate pairs generated:**
  - 46,794,809 test pairs, about 4.7 per S2/S3 record, from 142,179,392 before pruning.
  - With the chain-name step: 46,920,628.
  - On val: 15,972,076 pruned pairs.
- **How we ensured true matches were not lost:**
  - Recall is measured at real density (val queries against the full train S1 index), not on a small dev slice,
    where density inflates recall.
  - Channels are complementary: exact keys for clean duplicates, TF-IDF for typos and reordering, rare address
    words for name changes, and the Pass 7 anchor, skip-bigram and prefix channels.
  - Cosines are recomputed for every merged pair, so all channels compete fairly in pruning.
  - Key hits always survive pruning.
  - The pruning rule and n were chosen on the recall / F0.5-ceiling trade-off: pruned recall 0.9298 against 0.9361
    unpruned, at a third of the pairs.
  - The chain-name step recovers S1 that capped keys starve.

---

## 4. Matching Model

**Features used:**
- **Name features:**
  - rapidfuzz ratio, token_sort and token_set; Jaro-Winkler; partial ratio.
  - Word and character-trigram Jaccard on the legal-form-free core name.
  - Exact, first-word and acronym match; legal-suffix and DBA match; length ratio.
  - Pass 7 additions: brand-word disjointness, branch-modifier asymmetry, exact name with a missing address,
    space-free core ratio.
- **Address features:**
  - Token set / sort / partial ratio; word Jaccard; IDF-weighted shared address words.
  - Postcode match and 3-digit prefix; house-number Jaccard, first-number match and house-number conflict.
  - Landmark similarity; length ratio; missing-field count.
- **Other:**
  - The blocking evidence: eight key flags, two TF-IDF cosines, rare-word IDF and the pruning score.
  - **Record-relative context:** rank and gap to the record's best candidate on six scores, and its candidate
    count.
  - **S1-relative context:** candidate records pointing at the S1, and this record's rank among them.
  - Source flag.

**Model type:** LightGBM gradient-boosted trees: 3,902 trees, 63 leaves, negatives subsampled at 0.5, early
stopping on a record-level holdout.  
**Threshold selection method:** Corrected macro F0.5 on the eval half of the val candidate set. It is
precision-leaning by construction: F0.5, plus the 10x false-positive correction. It is applied after the
one-S1-per-record assignment. A joint study of extra decision rules (Pass 8) adopted none.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Eval corrected: 0.9531 (Pass 7), and 0.9561 with the chain-name step.
  - Eval uncorrected: 0.9563 / 0.9593.
  - Public leaderboard: 0.897 (chainfix, best), 0.8917 (Pass 7).
  - Final variant `chainfix`: leaderboard 0.897.

### Results across passes

| pass | main change | eval F0.5 | eval F0.5 corrected | pair recall | public leaderboard |
|---|---|---|---|---|---|
| 4 | LightGBM baseline on the val split, decision layer | 0.9274 | 0.9170 | - | 0.7819 |
| 5 | Team normalization and 27 pair features, negatives at 0.5 | 0.9345 | 0.9302 | 0.8729 | 0.8479 |
| 7 | Noise-hardened normalization, 5 new features, 3 new blocking channels, rescored pruning | 0.9563 | 0.9531 | 0.9109 | **0.8917** |
| 8 | Decision-rule study (no rule adopted; output = Pass 7) | 0.9563 | 0.9531 | 0.9109 | - |
| 8c | Chain-name candidate step (`chainfix`) | 0.9593 | 0.9561 | 0.9127 | **0.897** |
| 8c | `chainfix_t085` / `t085` / `t090` (stricter t) | 0.9584 / 0.9554 / 0.9536 | 0.9558 / 0.9528 / 0.9518 | - | not submitted |
| **final** | **`chainfix`** | | **0.9561** | | **0.897** |

Per country at Pass 7, corrected: India 0.9209, US 0.9745. With the chain-name step: India 0.9279, US 0.9748.
Singleton accuracy is 0.993.

### Error analysis (Pass 7, eval side, t = 0.80)

- **Common false negatives (missed matches).** In a sample of 3,000 false-negative pairs:
  - **79% never became candidates.** This is the dominant loss. Blocking recall caps F0.5 at 0.9659. Typical
    cases: heavily reworded names, names hidden behind URLs or transliteration variants, and generic chain names
    starved by key caps (partly fixed by the chain-name step).
  - 15% scored below t. These are mostly true duplicates with a perturbed house number (the generator changes
    numbers: 94-96% of accepted house-number-conflict pairs are true) or a missing address.
  - 6% were assigned to a different S1 of the same chain.
- **Common false positives (wrong merges).** 1,917 raw false positives:
  - **Sibling businesses:** same street, neighbouring house number, name = the S1 name plus a word like "Holding",
    "International", "Développement" or "Groupe".
  - **Name-only records** (empty address) whose chain name fits dozens of S1.
  - Same-name branches in the same city.

  In the 0.3-0.8 probability band these patterns are true only 26-47% of the time (x10 corrected). The model is
  calibrated on them, and rejecting them is right for F0.5.

### Leaderboard gap and the France investigation

Eval and leaderboard disagree by 0.14 (Pass 4), 0.08 (Pass 5) and 0.06 (Pass 7). Leaderboard gains ran about
twice the eval gains (Pass 5: +0.066 against +0.013; Pass 7: +0.044 against +0.023), consistent with France,
absent from eval, benefiting most from normalization work. The investigation (Pass 6 and 8b) found the following:
- **No pipeline bug.** The test path reproduces eval exactly (parity 2e-13). Ids round-trip losslessly. Country
  strings are exact. Every validator passes. A real bug was found and fixed: Git's `core.autocrlf` rewrote
  `models/lgbm.txt` to CRLF, which LightGBM cannot load. It is fixed with `.gitattributes: models/*.txt -text`,
  and the submitted files were unaffected.
- **Test predictions look like eval in every country.** The France model output is the most confident of all: 82%
  of chosen pairs at p >= 0.999, and 63% exact core-name agreement against 43-52% elsewhere. Its match-count
  distribution follows train truth.
- **France's defect is blocking.** 4.1% of France S1 have no candidate (India 1.9%, US 0.1%), caused by
  chain-name crowding. This was addressed by the chain-name step.
- **Backing out France's score.** A diagnostic file with every France row emptied was prepared. With
  f = 0.150 and singleton rate s = 0.0558, France's own score is F = s + (L - D) / f from its leaderboard score D,
  and the rest is R = (D - f·s) / (1 - f).

  If the rest matched eval (0.953), France would be at about 0.54. France alone explains the gap only in that
  case. The bound R <= 1 gives F >= 0.28. A second gap source was also measured: test India and US accept
  2.5-3x more borderline pairs (p 0.8-0.9) than eval. That motivated the stricter-threshold variants.

---

## 6. Conclusion

A normalization -> multi-channel blocking -> LightGBM -> assignment pipeline reaches 0.953-0.956 corrected macro
F0.5 on a realistic held-out split and 0.897 on the public leaderboard. It runs end to end on a 16 GB laptop in
about 4 hours.

Three lessons:
- The largest single gain came from **record-relative context features**, which let the model choose among
  competing S1 instead of scoring pairs in isolation.
- The largest remaining loss is **blocking recall** (79% of missed pairs).
- **Validation must match test density and coverage.** Eval never saw France, and test candidate lists are
  denser. That is where the leaderboard gap lives.

### Engineering for scale on a 16 GB laptop

- **Streaming and partitioning.** Records are processed per country in 200k-record parts, and each part holds
  whole records, so all record-level context is local. Norm records are loaded 8 parts at a time. Pair
  predictions go to disk per part.
- **Compact ids.** Every entity id becomes a lossless int64 key (`S<source>-<digits>`), which avoids string
  joins over 142M pairs.
- **Fast retrieval.** Prefix-filtered sparse top-k (`sparse_dot_topn`) replaces a full sparse product that would
  take hours. The rare-word retriever bounds its postings by construction.
- **Robustness.** An in-process memory guard exits cleanly (code 3) when free RAM drops below 1.5 GB. Every
  output is written to `.tmp` and renamed. Every stage is resumable: blocking parts, feature parts, and test
  predictions keyed by the SHA-256 of the model, `features.py` and `normalize.py`. A driver runs the whole pass
  overnight.
- **Validation at scale.** The organizer validator needs more than 7.5 GB for the 47M-id candidate file, so
  `src.check_candidates` applies the same rules streamed (1.2-2.1 GB).
- **Measured peaks:** blocking 5.3-6.5 GB, feature building 4.0 GB, training 4.2 GB, test scoring 4.0 GB.

### Limitations and next steps

- **Blocking recall (0.93) is the ceiling.** Next steps: extend the chain-name join to crowded names that do have
  some candidates, add phonetic and transliteration-variant keys for Indic names, and learn the pruning score.
- **No France labels.** Every France decision is extrapolated from India and US. Pseudo-labelling France with
  high-confidence pairs, or France-specific normalization such as dotted legal forms ("S.A.S" left as `s a s`,
  2.85% of France records), are cheap next steps.
- **Density shift between eval and test.** A density-aware calibration, or eval with the other records fully
  queried, would align the threshold with test.
- **Global threshold.** Per-country or per-density thresholds could not be validated for France. The Pass 8 study
  found no gain from local decision rules on eval.
- **Graph constraints.** A transitivity-based cleanup between S2 and S3 clusters was not explored.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains:
- `src/`: 17 modules.
- `tests/`: 71 unit tests.
- `models/`: `lgbm.txt`, `threshold.json`, `decision.json`, `train_summary.json`, byte-identical to the models
  that produced the outputs.
- `utils/validate_submission.py`: the organizer validator.
- `README.md`, `requirements.txt` (exact pins, Python 3.11).

Entry points, in order:
1. `python -m src.data`
2. `python -m src.split`
3. `python -m src.norm_cache train test`
4. `python -m src.candsets val`
5. `python -m src.valsplit`
6. `python -m src.matcher features val`
7. `python -m src.matcher train`
8. `python -m src.postprocess sweep`
9. `python -m src.blocking test`
10. `python -m src.run_pipeline test`

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The chain-name variants add
`python -m src.chainfix test france|india` and `python -m src.variants chainfix <t> <name>`.

A fast path uses the included model: `norm_cache test`, `blocking test`, then `run_pipeline test`, about 2.7
hours. The README lists the time and peak memory of every stage and the licenses of all dependencies (MIT, BSD,
Apache-2.0, ISC).

A packaging smoke test ran this code in a fresh virtual environment built from `requirements.txt`, on a
5,000-S1 test slice. It reproduced the corresponding rows of the submitted Pass 7 files exactly. No external data,
APIs or lookups are used.

### B. Additional Results

**Threshold sweep, Pass 7 eval side:**

| t | F0.5 | F0.5 corrected | precision | precision corrected | recall | singleton acc |
|---|---|---|---|---|---|---|
| 0.50 | 0.9571 | 0.9494 | 0.9939 | 0.9493 | 0.9201 | 0.9829 |
| 0.64 (uncorrected optimum) | 0.9572 | - | - | - | - | - |
| 0.80 | 0.9563 | **0.9531** | 0.9973 | 0.9776 | 0.9109 | 0.9929 |
| 0.85 | 0.9554 | 0.9528 | 0.9978 | 0.9821 | 0.9075 | 0.9944 |
| 0.90 | 0.9536 | 0.9518 | 0.9984 | 0.9868 | 0.9021 | 0.9957 |

**Test output profile (Pass 7, t = 0.80):**

| | France | India | US |
|---|---|---|---|
| S1 | 259,452 | 809,986 | 663,106 |
| S1 with an empty list | 7.98% | 7.85% | 4.70% |
| matches per S1 | 3.39 | 3.22 | 3.50 |
| chosen pairs at p >= 0.999 | 82.1% | 73.0% | 77.3% |
| matched pairs | 878,932 | 2,604,346 | 2,319,456 |

**Blocking recall by country (val, pruned n = 3):** India 0.8784, US 0.9642.
