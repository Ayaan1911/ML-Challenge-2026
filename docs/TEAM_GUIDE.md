Amazon ML Challenge 2026

**Team Build Guide: Business Entity Resolution**

Shared context for MA, AM and RA and for their AI assistants. Version 1.1 (changes: docs/GUIDE_UPDATES.md),
Friday 25 Sept 2026. Deadline: Sunday 27 Sept 2026, 11:59 PM IST.

0\. How to use this document

This guide is written for two readers at once: the three of us, and the
AI assistant each of us is working with. Everything an AI needs to write
correct code for its part is in here, including exact function names,
inputs, outputs and test examples.

If you are a human (AM or RA)

1.  Read Sections 1 to 5 once (about 15 minutes). Then read your own
    task card in Section 6.

2.  Upload this whole document to your AI chat, then paste your starter
    prompt from Appendix A (AM) or Appendix B (RA).

3.  Free-tier chats forget things or hit limits. If you start a new
    chat, upload the document again and paste the starter prompt again.
    If upload fails, paste Section 5 plus your task card as text.

4.  If your AI suggests renaming a function, changing arguments, adding
    a library or editing a file you do not own, say no. Section 5 is the
    source of truth. Ask MA if something seems wrong.

If you are an AI assistant reading this

- You are helping one team member build exactly one module of a larger
  pipeline. Other people write the other modules at the same time, so
  the interface in Section 5 must be followed character for character:
  same file path, function names, argument order, return types and
  dictionary keys.

- Do not invent extra requirements, do not restructure the project, and
  do not touch files outside your owner's task card.

- Section 9 lists hard rules from the competition. Breaking them (for
  example calling any internet API) disqualifies the whole team.

- When unsure, write the simplest version that satisfies the contract
  and the tests, and tell your user what you assumed.

1\. Team and ownership

| **Code** | **Role**                                                                     | **Tools**                         | **Owns (only these files)**                                                                    |
|----------|------------------------------------------------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------|
| MA       | Team lead, integrator. Owns the main branch and all leaderboard submissions. | Claude Code, laptop               | Everything in src/ except normalize.py and features.py. Final package, submissions, merging.   |
| AM       | Text normalization engineer                                                  | AI chat (any), laptop with Python | src/normalize.py, tests/test_normalize.py. Later: error analysis notes, reproducibility check. |
| RA       | Similarity features engineer                                                 | AI chat (any), laptop with Python | src/features.py, tests/test_features.py. Later: README.md and methodology document draft.      |

Why this split: MA has Claude Code and can drive the large,
interconnected parts of the pipeline. AM and RA each own one
self-contained module with a fixed interface. A chat AI can build such a
module well without seeing the rest of the code, and MA can plug it in
without rewriting anything.

2\. The problem in plain words

We get business records from **3 sources**. Each record has a name, an
address and a country, all noisy. Nothing links the sources together;
there is no shared ID. **Source 1 (S1)** is a clean reference list with
no duplicates. **Source 2 (S2)** and **Source 3 (S3)** are messy lists
that may describe the same businesses in different words.

**Our task:** for every S1 business, list every S2 and S3 record that is
the same real-world business. That can be zero, one or many records.
This task is called **entity resolution (ER)** or record linkage.

2.1 Data files (all tab-separated .tsv)

| **File**                                                      | **Contents**                                                                                                    |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| dataset/train/train_source1.tsv, \_source2.tsv, \_source3.tsv | Training records. Columns: entity_id, business_name, business_address, country                                  |
| dataset/train/train_ground_truth.tsv                          | Answers for training. Columns: source1_entity_id, matched_entity_ids (comma-separated S2/S3 ids, empty if none) |
| dataset/test/test_source1.tsv, \_source2.tsv, \_source3.tsv   | Test records, same columns, no answers. We predict for every S1 test record.                                    |

entity_id prefix tells the source: S1-, S2-, S3-. Always read files with
sep="\t", dtype=str and keep_default_na=False so empty cells stay empty
strings.

2.2 The noise we must handle

- **Names:** abbreviations (Corp / Corporation, Pvt / Private, Ltd /
  Limited), legal suffixes present or missing, DBA / trade names, &
  versus "and", word order swapped, typos, transliterations.

- **Addresses:** Rd / Road, St / Street, missing PIN code or state,
  landmarks such as "Near SBI ATM", different house-number formats,
  parts reordered.

- **Countries:** training has only US and India. **The test set also has
  France**, which never appears in training. Our code must treat country
  as any string and still work for France.

2.3 Outputs we submit

- output/matching_results.tsv: columns source1_entity_id,
  matched_entity_ids. Exactly one row per S1 test entity, ids
  comma-separated, empty if no match. This is the scored file.

- output/candidate_pairs.tsv: columns source1_entity_id,
  candidate_entity_ids. The shortlist our model scored for each S1
  entity. Every matched id must also be a candidate.

- A zip with code, a README, requirements.txt and a filled methodology
  document.

2.4 How we are scored (the most important part)

Score = **F0.5 per S1 entity, averaged over all S1 entities**. F0.5
weights precision (being right about what you claim) twice as much as
recall (finding everything).

> F0.5 = (1.25 x Precision x Recall) / (0.25 x Precision + Recall)
>
> Precision = correct predicted ids / predicted ids
>
> Recall = correct predicted ids / true ids
>
> Special cases: true empty & predicted empty -\> 1.0
>
> true empty & predicted anything -\> 0.0
>
> true non-empty & predicted empty -\> 0.0

| **Truth for one S1** | **We predict**         | **Score for that S1** |
|----------------------|------------------------|-----------------------|
| no matches           | nothing                | 1.0 (free point)      |
| no matches           | one wrong id           | 0.0                   |
| A, B, C              | A                      | 0.71                  |
| A, B, C              | A, B, C, X (one wrong) | 0.79                  |
| A                    | X (wrong)              | 0.0                   |

> **Lesson:** a wrong match costs more than a missed one, and correctly
> saying "no match" earns a full point. When the model is unsure, leave
> the id out.

2.5 Competition rules that affect code

- **No external data lookups:** no APIs, no geocoding, no business
  registries, no internet data. Disqualification if found. Hand-written
  rules and abbreviation tables are fine.

- The final model must be MIT or Apache 2.0 licensed and at most 8
  billion parameters.

- Do not hard-code, filter or one-hot on the set {US, India}. Every test
  entity, France included, must appear in the output.

3\. The solution we are building

A standard, proven entity-resolution pipeline in four stages. Records
flow top to bottom; each stage lives in one file with one owner.

| **Stage**          | **What it does**                                                                                           | **File**                                | **Owner** |
|--------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------|-----------|
| 0\. Load and split | Read TSVs; hold out 20% of training S1 entities as a validation set                                        | src/data.py, src/split.py               | MA        |
| 1\. Normalize      | Clean every name and address into a standard form; pull out postcode, numbers, landmark, legal suffix      | src/normalize.py                        | AM        |
| 2\. Blocking       | For each S1, pick a shortlist (about 20 to 50) of plausible S2/S3 records using fast search                | src/blocking.py                         | MA        |
| 3a. Pair features  | For one (S1, candidate) pair, compute about 27 similarity numbers                                          | src/features.py                         | RA        |
| 3b. Model          | Add corpus-level features (TF-IDF, embeddings, ranks); a LightGBM model gives a match probability per pair | src/matcher.py                          | MA        |
| 4\. Decide         | Keep each S2/S3 record only for its best S1; apply a threshold tuned for F0.5; write both TSVs             | src/postprocess.py, src/run_pipeline.py | MA        |
| Evaluate           | Exact competition metric on the validation split                                                           | src/metric.py                           | MA        |

In one line: raw record → normalize_record() → blocking gives candidate
pairs → pair_features(a, b) plus MA's extra features → LightGBM
probability → decision rules → output files.

**Where your module is used.** MA calls normalize_record() once for
every record in all six files, so it must be fast and must never crash.
Blocking and features then only read the NormRecord fields. MA calls
pair_features() for every candidate pair, possibly a few million times,
so it must be fast too.

**Stubs on day one.** MA will commit simple stub versions of
normalize.py and features.py with the exact Section 5 signatures, so the
pipeline runs end to end immediately. AM's and RA's real versions
replace the stubs file-for-file. Because the interface is identical,
integration means swapping files and re-running.

4\. Repository, setup and Git rules

4.1 Layout (matches the required submission package)

> business_entity_resolution/ \<- GitHub repo root
>
> src/
>
> \_\_init\_\_.py
>
> data.py split.py metric.py (MA)
>
> normalize.py (AM)
>
> features.py (RA)
>
> blocking.py matcher.py postprocess.py run_pipeline.py (MA)
>
> tests/
>
> test_normalize.py (AM)
>
> test_features.py (RA)
>
> docs/ team guide, error-analysis notes
>
> dataset/ train/ test/ (NOT committed, in .gitignore)
>
> output/ (NOT committed, in .gitignore)
>
> README.md (RA drafts, MA finalizes)
>
> requirements.txt (MA)

4.2 One-time setup (AM and RA)

> git clone \<REPO_URL_FROM_MA\>
>
> cd business_entity_resolution
>
> python -m venv .venv
>
> \# Windows: .venv\Scripts\activate Mac/Linux: source
> .venv/bin/activate
>
> pip install -r requirements.txt
>
> \# copy the dataset folder you downloaded into
> business_entity_resolution/dataset/
>
> git checkout -b am-normalize \# AM
>
> git checkout -b ra-features \# RA
>
> python -m pytest -q \# should run (stub tests may be empty)

4.3 Daily Git workflow

> \# after editing ONLY your own files
>
> python -m pytest -q tests/test_normalize.py \# AM (RA:
> tests/test_features.py)
>
> git add src/normalize.py tests/test_normalize.py \# never "git add ."
>
> git commit -m "normalize: v1 name + address rules"
>
> git push -u origin am-normalize \# RA: ra-features
>
> \# then message the group: "pushed am-normalize \<first 7 chars of
> commit hash\>, tests pass"

Git rules

- **Never push to main.** Only MA merges into main.

- Only commit the files listed for you in Section 1. Never commit
  dataset/, output/, .venv/ or notebooks with data printed in them.

- Commit small and often. A pushed working version at every checkpoint
  beats a perfect version that arrives late.

- If MA asks you to update from main: git fetch origin then git merge
  origin/main. Do not rebase.

4.4 Looking at the data yourself (recommended, 20 minutes)

Run this from the repo root to see real matched groups. Paste 20 to 30
rows of output into your AI chat when you want it to learn the noise
patterns. Never paste whole files; free chats will choke.

> import pandas as pd
>
> rd = lambda p: pd.read_csv(p, sep="\t", dtype=str,
> keep_default_na=False)
>
> s1 = rd("dataset/train/train_source1.tsv").set_index("entity_id")
>
> oth = pd.concat(\[rd("dataset/train/train_source2.tsv"),
>
> rd("dataset/train/train_source3.tsv")\]).set_index("entity_id")
>
> gt = rd("dataset/train/train_ground_truth.tsv")
>
> print("countries:", s1\["country"\].value_counts().to_dict())
>
> print("singleton share:", (gt.matched_entity_ids == "").mean())
>
> for \_, g in gt\[gt.matched_entity_ids != ""\].sample(15,
> random_state=0).iterrows():
>
> r = s1.loc\[g.source1_entity_id\]
>
> print("S1:", r.business_name, "\|", r.business_address, "\|",
> r.country)
>
> for m in g.matched_entity_ids.split(","):
>
> o = oth.loc\[m\]
>
> print(" ", m, ":", o.business_name, "\|", o.business_address, "\|",
> o.country)
>
> print()

5\. Interface contracts (source of truth)

> Everything in this section is fixed. Names, argument order, return
> types and dictionary keys must match exactly. Internal helper
> functions (names starting with \_) are free to add.

5.1 Rules for both modules

- Python 3.10 or newer. Allowed imports: Python standard library, numpy,
  pandas, rapidfuzz. Nothing else without MA's approval.

- No network access, no file reading or writing, no printing inside
  library functions.

- Pure and deterministic: same input gives the same output every time.

- Never raise on bad input. None, float NaN, empty string and unexpected
  countries must all work and return sensible empty values.

- Country is an open set. Apply special rules when a country string
  matches (case-insensitive, for example contains "india", or equals
  "us", "usa", "united states", "france", "fr"), and fall back to
  generic rules for anything else. Never reject or drop a record because
  of its country.

5.2 src/normalize.py (owner AM)

> """src/normalize.py - owner AM. Stdlib only: re, unicodedata, typing,
> math."""
>
> def normalize_country(country) -\> str:
>
> """Lowercase and strip. None/NaN -\> ''. Never maps to a fixed
> list."""
>
> def clean_text(text) -\> str:
>
> """Generic cleaner used by everything else.
>
> None/NaN -\> ''. Strip accents (NFKD, drop combining marks: 'Elysees'
> from 'Élysées').
>
> Lowercase. '&' -\> ' and '. '@' -\> ' at '.
>
> Every character that is not a letter, digit or space becomes a space
>
> (so hyphens, slashes, dots, commas, '#', quotes all become spaces).
>
> Collapse repeated spaces, strip ends."""
>
> def split_dba(name, country: str = "") -\> tuple\[str, str\]:
>
> """Works on the RAW name. Splits on (case-insensitive) 'd/b/a',
> 'd.b.a.',
>
> 'dba', 'doing business as', 'trading as', 't/a', 'aka', 'a.k.a.'.
>
> Returns (clean_text(primary), clean_text(trade_name)).
>
> No DBA marker -\> (clean_text(name), '')."""
>
> def split_legal_suffix(name_clean: str, country: str = "") -\>
> tuple\[str, str\]:
>
> """Input: output of clean_text. Repeatedly strips legal-form words
> (Table 5.3)
>
> from the END of the name. The French forms (sarl, sas, sasu, sa, eurl,
> snc,
>
> sci, scop) are ALSO stripped from the START, because French names
> often put
>
> them first. Also strips a leading 'the'.
>
> Returns (core_name, canonical_suffix). canonical_suffix = canonical
> forms
>
> from Table 5.3 joined by one space in original order, '' if none.
>
> Never returns an empty core if the name had other words: if stripping
> would
>
> leave nothing, keep the original words as the core."""
>
> def normalize_name(name, country: str = "") -\> str:
>
> """clean_text(name) with every word replaced by its canonical form
> from
>
> Table 5.3 (legal forms) where one exists. Keeps ALL words including a
> DBA
>
> part (the DBA marker itself becomes the single word 'dba')."""
>
> def normalize_address(address, country: str = "") -\> str:
>
> """clean_text(address) with each word (or listed multi-word phrase)
> replaced
>
> by its canonical form from Table 5.4. Removes the word 'cedex'.
>
> Keeps word order, numbers, postcode and landmark words."""
>
> def extract_postcode(address, country: str = "") -\> str:
>
> """Digits only, or ''. Works on the RAW address.
>
> india -\> 6 digits, first digit 1-9 ('560 041' with one space also
> counts -\> '560041')
>
> us/usa/united states -\> 5 digits (drop any -1234 extension)
>
> france/fr -\> 5 digits
>
> anything else -\> a standalone 5- or 6-digit run
>
> If several candidates, return the LAST one in the string."""
>
> def extract_numbers(address, country: str = "") -\> list\[str\]:
>
> """All digit runs in the address except the postcode, in order of
> appearance,
>
> duplicates removed. Leading zeros stripped ('004' -\> '4'). Ordinals
> reduced
>
> ('4th' -\> '4', '2nd' -\> '2', '1er' -\> '1'). Empty list if none."""
>
> def extract_landmark(address, country: str = "") -\> str:
>
> """Works on the RAW address. Finds cue words (case-insensitive, whole
> words):
>
> near, nr, opp, opposite, behind, beside, next to, adjacent to, in
> front of,
>
> pres de, face a (accents ignored). Takes the text after the cue up to
> the next
>
> comma or end of string, cleans it, drops the cue word.
>
> Several landmarks -\> joined with ' \| '. None -\> ''."""
>
> def normalize_record(record: dict) -\> dict:
>
> """MAIN ENTRY POINT used by MA. Input keys: entity_id, business_name,
>
> business_address, country (values may be None/NaN/missing).
>
> Returns a NormRecord dict with exactly the keys of Table 5.2."""

Table 5.2: NormRecord keys (returned by normalize_record)

| **Key**      | **Type**    | **Definition**                                                                                                      |
|--------------|-------------|---------------------------------------------------------------------------------------------------------------------|
| entity_id    | str         | Unchanged input id ('' if missing)                                                                                  |
| country_norm | str         | normalize_country(country)                                                                                          |
| name_norm    | str         | normalize_name(business_name, country)                                                                              |
| name_core    | str         | split_legal_suffix(primary part of split_dba(name))\[0\], with Table 5.3 canonical forms applied to remaining words |
| legal_suffix | str         | split_legal_suffix(...)\[1\]                                                                                        |
| dba_name     | str         | Trade name from split_dba, legal suffix stripped the same way; '' if none                                           |
| addr_norm    | str         | normalize_address(business_address, country)                                                                        |
| addr_core    | str         | addr_norm with the postcode, the landmark text and the cue words removed, spaces collapsed                          |
| postcode     | str         | extract_postcode(business_address, country)                                                                         |
| numbers      | list\[str\] | extract_numbers(business_address, country)                                                                          |
| landmark     | str         | extract_landmark(business_address, country), then normalize_address applied to it                                   |

Table 5.3: Legal forms and name words (variant → canonical)

| **Canonical**                             | **Variants (after clean_text)**                              | **Typical country** |
|-------------------------------------------|--------------------------------------------------------------|---------------------|
| inc                                       | inc, incorporated                                            | US                  |
| corp                                      | corp, corporation                                            | US                  |
| co                                        | co, company                                                  | any                 |
| llc                                       | llc, l l c                                                   | US                  |
| llp                                       | llp, l l p                                                   | US, India           |
| lp                                        | lp                                                           | US                  |
| plc                                       | plc                                                          | any                 |
| ltd                                       | ltd, limited                                                 | any                 |
| pvt                                       | pvt, private, pvte                                           | India               |
| opc                                       | opc                                                          | India               |
| sarl, sas, sasu, sa, eurl, snc, sci, scop | same words (canonical = itself)                              | France              |
| and                                       | and, & (via clean_text), et (French, only between two words) | any                 |

Example: "Private Limited" becomes "pvt ltd"; "Corporation" becomes
"corp". AM may **add** rows when the data shows new patterns, but must
not remove or rename existing canonical forms (MA's code and RA's tests
rely on them).

Table 5.4: Address words (variant → canonical)

| **Canonical** | **Variants**              | **Canonical** | **Variants**                |
|---------------|---------------------------|---------------|-----------------------------|
| st            | street, str, st           | rd            | road, rd                    |
| av            | avenue, ave, av, avn      | bd            | boulevard, blvd, boul, bd   |
| ln            | lane, ln                  | dr            | drive, dr                   |
| ct            | court, ct                 | pl            | place, pl, plz, plaza       |
| hwy           | highway, hwy              | rte           | route, rte                  |
| ste           | suite, ste                | apt           | apartment, apt              |
| fl            | floor, flr, fl            | bldg          | building, bldg, bldng       |
| no            | number, num, no           | n / s / e / w | north / south / east / west |
| near          | near, nr                  | opp           | opposite, opp               |
| cross         | cross, crs                | main          | main, mn                    |
| sector        | sector, sec               | mg            | mahatma gandhi              |
| ch            | chemin                    | imp           | impasse                     |
| all           | allee                     | fbg           | faubourg                    |
| bangalore     | bengaluru, bangalore, blr | mumbai        | mumbai, bombay              |
| chennai       | chennai, madras           | kolkata       | kolkata, calcutta           |
| gurgaon       | gurugram, gurgaon         | new delhi     | new delhi, delhi, nd        |

Some collisions are accepted (French "Ste" = Sainte and "St" = Saint
also map to ste and st). Both sides of a pair get the same treatment, so
matching still works.

5.3 src/features.py (owner RA)

> """src/features.py - owner RA. Imports allowed: stdlib, numpy, pandas,
> rapidfuzz."""
>
> from rapidfuzz import fuzz
>
> from rapidfuzz.distance import JaroWinkler
>
> FEATURE_NAMES: list\[str\] = \[ ...exactly the 27 names of Table 5.5,
> in that order... \]
>
> def pair_features(a: dict, b: dict) -\> dict\[str, float\]:
>
> """a = NormRecord of the Source 1 record, b = NormRecord of a
> candidate
>
> (Source 2 or 3). Returns a dict with EVERY key in FEATURE_NAMES, all
> Python
>
> floats. Similarities are in \[0, 1\] (divide rapidfuzz scores by 100).
>
> If a feature cannot be computed because a needed field is empty on
> either
>
> side, return -1.0 for it (never 0, never NaN). Must be symmetric:
>
> pair_features(a, b) == pair_features(b, a)."""
>
> def pair_features_batch(a_list: list\[dict\], b_list: list\[dict\])
> -\> list\[dict\[str, float\]\]:
>
> """Same as calling pair_features on zip(a_list, b_list). Simple loop
> is fine."""
>
> \# Added in interface v1.1 (see docs/GUIDE_UPDATES.md). Imports
> allowed now also include pandas.
>
> def pair_features_frame(a: pd.DataFrame, b: pd.DataFrame) -\>
> pd.DataFrame:
>
> """a, b = row-aligned DataFrames whose columns are the NormRecord keys
> (row i of a
>
> is paired with row i of b, by position). Returns a DataFrame with
> len(a) rows and
>
> float32 columns named exactly FEATURE_NAMES, in order, with the SAME
> values as
>
> pair_features on each row pair (tolerance 1e-5), including the -1.0
> rule.
>
> Must be vectorized: rapidfuzz.process.cpdist(..., workers=-1) for
> string scores
>
> plus numpy for the rest. No Python loop per row. Target: at least 1M
> pairs per
>
> minute on a laptop. pair_features stays as the per-pair reference
> implementation."""
>
> MA's tests/test_interface.py checks the names, the parity on 1,000
> random pairs and the speed.

Table 5.5: Features (in this exact order)

| **\#** | **Name**           | **Definition (a, b are NormRecords)**                                                                                                                            |
|--------|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1      | name_ratio         | fuzz.ratio(name_norm) / 100                                                                                                                                      |
| 2      | name_token_sort    | fuzz.token_sort_ratio(name_norm) / 100                                                                                                                           |
| 3      | name_token_set     | fuzz.token_set_ratio(name_norm) / 100                                                                                                                            |
| 4      | core_jw            | JaroWinkler.similarity(name_core)                                                                                                                                |
| 5      | core_token_set     | fuzz.token_set_ratio(name_core) / 100                                                                                                                            |
| 6      | core_partial       | fuzz.partial_ratio(name_core) / 100                                                                                                                              |
| 7      | core_exact         | 1.0 if name_core equal, else 0.0                                                                                                                                 |
| 8      | core_word_jaccard  | Jaccard of the word sets of name_core                                                                                                                            |
| 9      | core_char3_jaccard | Jaccard of character 3-gram sets of name_core with spaces removed                                                                                                |
| 10     | first_word_match   | 1.0 if first word of name_core equal, else 0.0                                                                                                                   |
| 11     | acronym_match      | 1.0 if acronym (first letters of words) of one core, length \>= 2, equals the other core with spaces removed; or both acronyms equal with length \>= 3; else 0.0 |
| 12     | legal_suffix_match | 1.0 equal, 0.0 different, -1.0 if either empty                                                                                                                   |
| 13     | dba_best           | max fuzz.token_set_ratio / 100 over all pairs from {name_core, dba_name} of a and of b, ignoring empty strings                                                   |
| 14     | name_len_ratio     | min(len) / max(len) of name_core                                                                                                                                 |
| 15     | addr_token_set     | fuzz.token_set_ratio(addr_norm) / 100                                                                                                                            |
| 16     | addr_token_sort    | fuzz.token_sort_ratio(addr_core) / 100                                                                                                                           |
| 17     | addr_partial       | fuzz.partial_ratio(addr_core) / 100                                                                                                                              |
| 18     | addr_word_jaccard  | Jaccard of the word sets of addr_core                                                                                                                            |
| 19     | postcode_match     | 1.0 equal, 0.0 different, -1.0 if either empty                                                                                                                   |
| 20     | postcode_prefix3   | 1.0 if first 3 digits equal, 0.0 if not, -1.0 if either empty                                                                                                    |
| 21     | num_jaccard        | Jaccard of the numbers lists as sets; -1.0 if either empty                                                                                                       |
| 22     | first_num_match    | 1.0 if numbers\[0\] equal, 0.0 if not, -1.0 if either empty                                                                                                      |
| 23     | landmark_sim       | fuzz.token_set_ratio(landmark) / 100; -1.0 if either empty                                                                                                       |
| 24     | addr_len_ratio     | min(len) / max(len) of addr_core                                                                                                                                 |
| 25     | country_match      | 1.0 equal, 0.0 different, -1.0 if either empty                                                                                                                   |
| 26     | name_in_addr       | 1.0 if name_core of either record (length \>= 4) appears inside the other's addr_norm as whole words, else 0.0                                                   |
| 27     | missing_count      | number of empty fields among name_core, addr_core, postcode on both sides combined (0 to 6), as float                                                            |

For features 1 to 18, 24 and 26: if the string(s) the feature needs are
empty on either side, return -1.0. RA may add extra features at the
**end** of the list after telling MA, never in the middle or with
renamed keys.

5.4 Worked examples (use these as test fixtures)

These are the exact expected normalize_record outputs. AM's tests must
assert them. RA can use the outputs directly as NormRecord inputs before
AM's module is ready.

> \# Example 1 (India)
>
> IN_1 = {"entity_id": "S1-00001", "business_name": "Sharma & Sons Pvt.
> Ltd.",
>
> "business_address": "No. 12, 4th Cross, Near SBI ATM, Jayanagar,
> Bengaluru 560041",
>
> "country": "India"}
>
> OUT_1 = {"entity_id": "S1-00001", "country_norm": "india",
>
> "name_norm": "sharma and sons pvt ltd", "name_core": "sharma and
> sons",
>
> "legal_suffix": "pvt ltd", "dba_name": "",
>
> "addr_norm": "no 12 4th cross near sbi atm jayanagar bangalore
> 560041",
>
> "addr_core": "no 12 4th cross jayanagar bangalore",
>
> "postcode": "560041", "numbers": \["12", "4"\], "landmark": "sbi atm"}
>
> \# Example 2 (US, with DBA)
>
> US_1 = {"entity_id": "S2-00047", "business_name": "ACME Corporation
> d/b/a Acme Tools",
>
> "business_address": "1200 N. Main St., Suite 4, Springfield, IL
> 62704-1234",
>
> "country": "US"}
>
> OUT_2 = {"entity_id": "S2-00047", "country_norm": "us",
>
> "name_norm": "acme corp dba acme tools", "name_core": "acme",
>
> "legal_suffix": "corp", "dba_name": "acme tools",
>
> "addr_norm": "1200 n main st ste 4 springfield il 62704 1234",
>
> "addr_core": "1200 n main st ste 4 springfield il 1234",
>
> "postcode": "62704", "numbers": \["1200", "4", "1234"\], "landmark":
> ""}
>
> \# Example 3 (France - unseen in training, must still work)
>
> FR_1 = {"entity_id": "S3-00812", "business_name": "SARL Boulangerie
> Dupont",
>
> "business_address": "12 bis Av. des Champs-Élysées, 75008 Paris",
>
> "country": "France"}
>
> OUT_3 = {"entity_id": "S3-00812", "country_norm": "france",
>
> "name_norm": "sarl boulangerie dupont", "name_core": "boulangerie
> dupont",
>
> "legal_suffix": "sarl", "dba_name": "",
>
> "addr_norm": "12 bis av des champs elysees 75008 paris",
>
> "addr_core": "12 bis av des champs elysees paris",
>
> "postcode": "75008", "numbers": \["12"\], "landmark": ""}
>
> \# Example 4 (garbage input must not crash)
>
> BAD = {"entity_id": "S2-99999", "business_name": None,
>
> "business_address": float("nan"), "country": "Germany"}
>
> OUT_4 = {"entity_id": "S2-99999", "country_norm": "germany",
> "name_norm": "",
>
> "name_core": "", "legal_suffix": "", "dba_name": "", "addr_norm": "",
>
> "addr_core": "", "postcode": "", "numbers": \[\], "landmark": ""}

Note on Example 2: the ZIP extension 1234 is not the postcode, so it
stays in numbers and addr_core. That is acceptable and consistent on
both sides of a pair.

6\. Task cards

6.1 AM: normalization (src/normalize.py)

Phase 1 deliverable (Friday by 8 PM)

1.  Setup from Section 4.2, branch am-normalize. Run the data-viewing
    script in 4.4 and read 15 matched groups to get a feel for the
    noise.

2.  With your AI (starter prompt: Appendix A), write src/normalize.py
    implementing every function in 5.2 and the tables 5.3 and 5.4.

3.  Write tests/test_normalize.py: the four examples in 5.4 exactly,
    plus idempotence (normalize_name(normalize_name(x)) ==
    normalize_name(x)), accents, and an unknown country.

4.  Run a speed check: normalize_record on 50,000 records should take
    under about 20 seconds on your laptop.

5.  Run python -m pytest -q tests/test_normalize.py, push, and message
    the group with the commit hash.

Phase 2 (Saturday)

1.  MA shares docs/errors_val.csv: validation pairs the model got wrong,
    with names, addresses and scores.

2.  Read at least 60 rows. Group them by cause (for example: "S. N.
    Enterprises" versus "SN Enterprises", missing Indian city alias,
    landmark swallowed the street). Write the patterns into
    docs/am_error_notes.md with 2 examples each.

3.  Add rules or table rows that fix the common patterns, add a test for
    each, push. Target: Saturday 1 PM for v2.

4.  Build a France checklist: invent 15 realistic French name/address
    pairs (different formatting of the same business) and confirm
    normalize makes them close. Add them as tests.

Phase 3 (Sunday)

Reproducibility check: fresh clone of main into a new folder, follow
README.md exactly, run the full pipeline, confirm both output files are
produced and the validator prints PASS. Report every place the README
was unclear.

6.2 RA: pair features (src/features.py)

Phase 1 deliverable (Friday by 9 PM)

1.  Setup from Section 4.2, branch ra-features. Confirm import rapidfuzz
    works.

2.  With your AI (starter prompt: Appendix B), write src/features.py
    with FEATURE_NAMES and pair_features exactly as in 5.3 and Table
    5.5.

3.  Until AM's module lands, use the OUT\_ dictionaries from 5.4 as
    inputs. Your code must not import normalize.py.

4.  Write tests/test_features.py: every key present and a float; all
    values in \[-1, 1\] except missing_count in \[0, 6\]; symmetry on
    the example pairs; OUT_1 versus itself gives 1.0 on features 1 to 6
    and 15 to 17; an all-empty record gives -1.0 where specified and no
    crash; OUT_1 versus itself gives postcode_match 1.0; OUT_1 versus
    OUT_3 gives country_match 0.0.

5.  Speed test inside the tests: 20,000 calls to pair_features should
    finish in under about 10 seconds.

6.  Push and message the group with the commit hash.

Phase 2 (Saturday)

1.  MA shares feature importance from the LightGBM model and a sample of
    errors. Propose and implement up to 5 new features, appended at the
    end of FEATURE_NAMES, each with a test. Ideas: numeric-token overlap
    between names ("7 eleven"), longest common substring ratio,
    rare-word match. Push v2 by Saturday 2 PM.

2.  Draft README.md: purpose, setup, exact commands to reproduce (data →
    blocking → matching → output), expected runtime, where outputs land.
    MA gives you the final command lines.

Phase 3 (Sunday)

Fill Documentation_template.md (from the student resource) with the
methodology: pipeline overview, normalization rules, blocking strategy,
features table, model, threshold tuning, France handling, validation
results. MA supplies the numbers. Draft by Sunday 4 PM.

6.3 MA: core pipeline, integration, submissions

1.  Repo scaffold, requirements.txt, .gitignore, and **stubs** of
    normalize.py and features.py with the exact Section 5 signatures.
    Push main and share the repo link first thing.

2.  EDA facts to verify early: record counts per file, singleton share,
    matches per S1, whether any S2/S3 id appears in more than one
    ground-truth row (if never, enforce one S1 per record), whether
    matched pairs always share the same country label, S2 versus S3
    differences.

3.  split.py: 80/20 split by S1 entity, seed 42, saved to disk so all
    numbers are comparable. metric.py: exact competition F0.5 including
    the singleton rules.

4.  Baseline by Friday night: TF-IDF character n-gram blocking plus a
    threshold → first leaderboard submission.

5.  blocking.py: union of retrievers (name char TF-IDF top-k,
    name+address multilingual embeddings top-k, postcode/rare-token
    keys). Report candidate recall and average candidates per S1 on
    validation.

6.  matcher.py: RA's features + TF-IDF cosine + embedding cosine + rank
    features + source flag → LightGBM, grouped CV by S1.

7.  postprocess.py: one-S1-per-record rule (if EDA confirms), threshold
    sweep on validation macro F0.5, write both TSVs, run the validator.

8.  Integration: merge am-normalize and ra-features branches after their
    tests pass, rerun validation, share scores and errors_val.csv.

9.  Stretch: fine-tuned cross-encoder (xlm-roberta-base, MIT) ensembled
    with LightGBM. Freeze Sunday 6 PM; package and submit.

7\. Timeline (IST)

If the repo goes live later than planned, shift the Friday times by the
same delay but keep the Saturday and Sunday times fixed.

| **When**                   | **Who** | **Milestone**                                                                  |
|----------------------------|---------|--------------------------------------------------------------------------------|
| Fri, as soon as possible   | MA      | Repo + stubs pushed, link shared                                               |
| Fri, within 1 hour of link | AM, RA  | Setup done, branch created, data viewed                                        |
| Fri 8 PM                   | AM      | normalize.py v1 pushed, tests passing                                          |
| Fri 9 PM                   | RA      | features.py v1 pushed, tests passing                                           |
| Fri night                  | MA      | Baseline submission on leaderboard; v1 modules merged                          |
| Sat morning                | MA      | Full blocking + LightGBM; errors_val.csv shared                                |
| Sat 1 PM / 2 PM            | AM / RA | v2 modules pushed from error analysis                                          |
| Sat before midnight        | All     | Strong submission in before the 48-hour mark (top 500 get extra \$100 credits) |
| Sun 4 PM                   | RA      | README + methodology draft                                                     |
| Sun 6 PM                   | MA      | Model frozen; no new features after this                                       |
| Sun 6 to 9 PM              | AM, MA  | Clean re-run from scratch, validator PASS, zip built                           |
| Sun by 10 PM               | MA      | Final leaderboard upload + package submitted (2-hour buffer)                   |

8\. How integration works

1.  AM or RA pushes their branch and posts the commit hash in the group.

2.  MA pulls the branch, runs its tests, then runs the pipeline on
    validation with the new module.

3.  If the validation score holds or improves, MA merges into main and
    posts the new score.

4.  If something breaks, MA posts the failing input as a new test case.
    The owner adds it to their tests, fixes it on the same branch and
    pushes again. No one edits another person's file.

5.  Because stubs share the exact interface, merging a module is a file
    swap. If the interface ever needs to change, only MA changes it,
    updates this document, and tells everyone.

9\. Hard rules for every AI assistant

- No internet in the code: no APIs, no geocoding, no web lookups, no
  external datasets. This disqualifies the team.

- Do not hard-code or filter countries to US and India. France appears
  only in the test set.

- Do not change any name, signature, key or file path from Section 5.

- Do not add libraries beyond Section 5.1 without MA's approval.

- Only create or edit the files in your owner's task card.

- Any model used must be MIT or Apache 2.0 and at most 8B parameters
  (relevant to MA's code only).

10\. Glossary

| **Term**               | **Meaning**                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------|
| Entity resolution (ER) | Deciding which records describe the same real-world thing                                     |
| Blocking / candidates  | A fast first pass that picks a shortlist of possible matches, so we do not compare every pair |
| Singleton              | An S1 entity with no matches at all                                                           |
| Precision / recall     | Of what we predicted, how much is right / of what is true, how much we found                  |
| F0.5, macro-average    | Score that favors precision; computed per S1 then averaged                                    |
| Feature                | A number describing a pair (for example name similarity 0.92)                                 |
| LightGBM               | A fast tree-based model that learns how to combine features into a match probability          |
| Threshold              | Probability cut-off above which a pair is declared a match                                    |
| Validation split       | Training data we hide from the model to measure our score honestly                            |
| NormRecord             | The cleaned record dictionary produced by normalize_record (Table 5.2)                        |

Appendix A: AM starter prompts

Upload this document to your AI chat first, then paste the prompt.

**AM Pass 1: build src/normalize.py**

> I am AM, one of three members of a team in the Amazon ML Challenge
> 2026 (72-hour
>
> hackathon, Business Entity Resolution). I have attached our team
> guide; read it fully
>
> before answering. Sections 2 to 5 explain the problem and the exact
> interface contract.
>
> My job is ONLY src/normalize.py and tests/test_normalize.py (task card
> 6.1). My
>
> teammate MA will import my functions into the main pipeline, so every
> function name,
>
> argument, return type and NormRecord key must match Section 5 exactly.
> Do not rename
>
> anything, add required arguments, change return types, or touch other
> files.
>
> 1\. In at most 8 lines, restate what normalize.py must do and list
> every function
>
> signature from Section 5.2, so I can confirm you understood.
>
> 2\. Write the complete src/normalize.py using only the Python standard
> library
>
> (re, unicodedata, typing, math). Implement Tables 5.3 and 5.4 as
> module-level
>
> dictionaries. Handle None, float NaN and empty strings without
> raising.
>
> 3\. Write tests/test_normalize.py with pytest. Include the four
> examples of Section 5.4
>
> as exact equality tests, plus idempotence of normalize_name, accent
> stripping,
>
> the unknown country "Germany", and a speed test (50,000
> normalize_record calls
>
> in under about 20 seconds).
>
> 4\. Give me the exact commands to run the tests and to commit and push
> to my branch
>
> am-normalize (never main), adding only these two files.
>
> 5\. Never call any internet API, geocoding service or external
> dataset, and never
>
> hard-code the list of countries.

**AM Fix pass: when tests fail**

> Here is the pytest output from my laptop. Fix src/normalize.py so all
> tests pass.
>
> Keep every function signature and NormRecord key from Section 5 of the
> guide
>
> unchanged. If a test itself contradicts Section 5.4, tell me instead
> of changing it.
>
> 1\. Explain the cause of each failure in one line.
>
> 2\. Give me the full corrected file.
>
> \<paste pytest output here\>

**AM Pass 2: rules from error analysis**

> Below are rows from errors_val.csv: record pairs our model got wrong,
> with names,
>
> addresses, countries, the true label and the model score. Help me
> improve
>
> src/normalize.py (task card 6.1, phase 2) without changing any
> interface in Section 5.
>
> 1\. Group these errors into patterns caused by normalization, with 2
> examples each.
>
> Ignore errors normalization cannot fix.
>
> 2\. For each fixable pattern, propose a rule or a new table row (add
> rows only, never
>
> rename or remove canonical forms).
>
> 3\. Give me the updated normalize.py and one new test per rule.
>
> 4\. Draft docs/am_error_notes.md summarising the patterns.
>
> \<paste 40-60 rows here\>

Appendix B: RA starter prompts

Upload this document to your AI chat first, then paste the prompt.

**RA Pass 1: build src/features.py**

> I am RA, one of three members of a team in the Amazon ML Challenge
> 2026 (72-hour
>
> hackathon, Business Entity Resolution). I have attached our team
> guide; read it fully
>
> before answering. Sections 2 to 5 explain the problem and the exact
> interface contract.
>
> My job is ONLY src/features.py and tests/test_features.py (task card
> 6.2). My teammate
>
> MA will call pair_features millions of times inside the main pipeline,
> so
>
> FEATURE_NAMES, the function signatures and every feature key must
> match Section 5.3
>
> and Table 5.5 exactly, in the same order. Do not rename anything or
> touch other files.
>
> 1\. In at most 8 lines, restate what features.py must do, including
> the -1.0 rule for
>
> missing values and the symmetry requirement, so I can confirm you
> understood.
>
> 2\. Write the complete src/features.py using only the standard
> library, numpy and
>
> rapidfuzz. Do not import normalize.py; inputs are NormRecord dicts
> (Table 5.2).
>
> Never raise on empty strings or empty lists.
>
> 3\. Write tests/test_features.py with pytest, using the OUT_1, OUT_2,
> OUT_3 and OUT_4
>
> dictionaries from Section 5.4 as fixtures. Cover every check listed in
> task card
>
> 6.2, including a speed test (20,000 pair_features calls in under about
> 10 seconds).
>
> 4\. Give me the exact commands to run the tests and to commit and push
> to my branch
>
> ra-features (never main), adding only these two files.
>
> 5\. Never call any internet API or external dataset.

**RA Pass 2: new features**

> MA shared the LightGBM feature importance and example errors below.
> Help me add up
>
> to 5 new features to src/features.py (task card 6.2, phase 2).
>
> 1\. Propose features that would separate these errors, using only
> fields in the
>
> NormRecord (Table 5.2) and the allowed libraries.
>
> 2\. Append them to the END of FEATURE_NAMES; do not rename or reorder
> existing ones.
>
> 3\. Give me the updated file and one test per new feature. Keep
> symmetry and the
>
> -1.0 missing rule.
>
> \<paste importance table and error rows here\>

Appendix C: MA Claude Code passes

Put the Markdown copy of this guide at docs/TEAM_GUIDE.md in the repo so
Claude Code can read it.

**MA Pass 1: scaffold, EDA, metric, baseline**

> We are building an entity-resolution pipeline for the Amazon ML
> Challenge 2026. Read
>
> docs/TEAM_GUIDE.md fully first; Sections 2 to 5 and task card 6.3 are
> binding. Two
>
> teammates are writing src/normalize.py and src/features.py on other
> branches against
>
> the Section 5 interface, so those signatures must not change.
>
> 1\. Create the repo layout of Section 4.1 with .gitignore (dataset/,
> output/, .venv/)
>
> and requirements.txt pinning pandas, numpy, rapidfuzz, scikit-learn,
> lightgbm,
>
> pytest at the installed versions.
>
> 2\. Write stub src/normalize.py and src/features.py with the exact
> Section 5 names,
>
> signatures, NormRecord keys and FEATURE_NAMES, using simple logic.
>
> 3\. Write src/data.py (tab-separated, dtype=str,
> keep_default_na=False) and
>
> src/eda.py that prints every EDA fact listed in task card 6.3, step 2.
>
> 4\. Write src/split.py (80/20 by S1 entity, seed 42, saved to
> data/splits/) and
>
> src/metric.py implementing the exact macro F0.5 of Section 2.4, with
> unit tests
>
> for the special cases.
>
> 5\. Build a baseline: character n-gram TF-IDF on name + address for
> blocking,
>
> cosine threshold for matching, tuned on validation; write both TSVs to
> output/
>
> for the test set and run utils/validate_submission.py.
>
> 6\. Stop and report: EDA facts, candidate recall, validation F0.5,
> validator result.

**MA Pass 2: real blocking, LightGBM, decision rules**

> Continue from Pass 1, following docs/TEAM_GUIDE.md task card 6.3 steps
> 5 to 7.
>
> 1\. src/blocking.py: union of retrievers (name char TF-IDF top-k,
> name + address
>
> embeddings with a multilingual MIT/Apache sentence-transformer + FAISS
> top-k,
>
> exact postcode and rare-token keys). Report candidate recall and mean
> candidates
>
> per S1 on validation for several k values; pick k for recall of at
> least 97%.
>
> 2\. src/matcher.py: features = features.pair_features + TF-IDF
> cosine + embedding
>
> cosine + rank of the candidate within its S1 + rank of the S1 within
> the
>
> candidate + score gap to the next best + source flag. Train LightGBM
> with
>
> GroupKFold by S1 on the train split; predict validation.
>
> 3\. src/postprocess.py: if EDA showed each S2/S3 id maps to at most
> one S1, keep
>
> each candidate only for its best S1. Sweep the threshold to maximise
> validation
>
> macro F0.5.
>
> 4\. src/run_pipeline.py: one command that goes data -\> normalize -\>
> blocking -\>
>
> features -\> model -\> postprocess -\> output/ for the test set, then
> validates.
>
> 5\. Write docs/errors_val.csv (false positives and false negatives
> with names,
>
> addresses, countries, label, score) and a feature-importance table.
>
> 6\. Report validation F0.5 before and after each step.

**MA Pass 3: merge a teammate branch**

> My teammate pushed branch \<am-normalize \| ra-features\>. Integrate
> it safely.
>
> 1\. Fetch the branch, diff it against main, and confirm it only
> touches its owner's
>
> files and keeps the Section 5 interface exactly. List any violations.
>
> 2\. Run its tests and the full pytest suite.
>
> 3\. Run the pipeline on validation and compare macro F0.5 with main.
>
> 4\. If everything passes and the score does not drop, merge with
> --no-ff and push
>
> main. Otherwise, write the failing inputs as new test cases I can send
> back.
