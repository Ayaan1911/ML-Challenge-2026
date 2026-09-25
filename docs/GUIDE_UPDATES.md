# Guide updates: interface v1.1 (Friday 25 Sept)

Everything in TEAM_GUIDE.md still holds. This page lists what changed after
MA's first look at the real data, and what it means for each of you. Nothing
was renamed and no argument changed. There is one new function, for RA.

Why things changed: the data is much bigger than we planned for. The test set
has 1.73 million S1 records and about 10 million S2/S3 records. Blocking will
produce tens of millions of candidate pairs, so anything that runs once per
record or once per pair has to be fast.

## For both of you

- **Only 5.6% of S1 records have no match.** The other 94% have about 3.7
  matches on average (up to 11). The "no match gives a free point" rule is
  real, but it concerns only a few records. Do not tune anything to make "no
  match" more likely. Finding the true matches is where the score is.
- Facts you can rely on: every S2/S3 record belongs to at most one S1, and a
  matched pair always has the same country string.
- Run the whole test suite before you push: `python -m pytest -q`. It now
  includes MA's `tests/test_interface.py`, which checks the Section 5 names
  and keys against your module. If it fails, your branch cannot be merged.

## For RA (src/features.py)

**New required function** (Section 5.3 of the guide has been updated):

```python
def pair_features_frame(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
```

- `a` and `b` are DataFrames with the NormRecord keys as columns. Row *i* of
  `a` pairs with row *i* of `b`, by position; ignore the index labels.
- Return a DataFrame with `len(a)` rows and **float32** columns named
  exactly `FEATURE_NAMES`, in order.
- Every value must equal what `pair_features` gives for that row pair
  (tolerance 1e-5), including the -1.0 rule for empty fields.
- It must be **vectorized**: use `rapidfuzz.process.cpdist(x_list, y_list,
  scorer=..., workers=-1)` for the string scores and numpy for the rest
  (equality, Jaccard, lengths, counts). No Python `for` loop over rows. For
  the -1.0 rule, build a boolean mask of "both non-empty", score only those
  rows, and fill the rest with -1.0.
- **Speed target: at least 1 million pairs per minute** on your laptop.
- Keep `pair_features` exactly as it is. It is the slow, easy-to-read
  reference that the fast version is checked against.
- **Required test** in `tests/test_features.py`: build 1,000 random pairs
  (seeded), run both functions, and assert every column matches with
  `numpy.testing.assert_allclose(..., atol=1e-5)`. Also add a speed test of
  100,000 pairs in under about 6 seconds.
- The stub in `src/features.py` on main shows the pattern (`_vsim`, `_veq`),
  but it only computes 11 of the 27 features.
- pandas is now an allowed import in features.py.

## For AM (src/normalize.py)

**Speed:** `normalize_record` will be called about **22 million times** (all
six files), in 15 worker processes. Build every regex once at module level
with `re.compile(...)`. Never build a pattern inside a function; `re.sub(f"...{x}...")`
inside a call is the most common slowdown. Keep the tables as module-level
dicts and sets, not lists you search through. Target: 50,000 records in
under 20 seconds, as in the task card; faster is better.

**Noise the guide did not mention.** All examples are from
`docs/eda_report.md`. Each item is followed by the result we want.

1. **Non-Latin scripts.** Some names are written entirely in Devanagari. S1
   uses "Madhya Pradesh" while a match has "मध्य प्रदेश". A Hyderabad record has
   "తెలంగాణ" (Telangana). `clean_text` keeps these letters, which is correct:
   do not delete them. They will not match Latin text, though. A small table
   for Indian state names in native script that maps them to Latin
   (మధ్య ప్రదేశ్ / मध्य प्रदेश -> madhya pradesh, राजस्थान -> rajasthan,
   తెలంగాణ -> telangana, ...) would help addresses. A hand-written table is
   allowed; downloading one is not.
2. **Indian state abbreviations** appear next to full names: "RJ" vs
   "Rajasthan", "MP" vs "Madhya Pradesh", "MH" (Maharashtra), "DL" (Delhi).
   US states likewise: "TN" vs "Tennessee", "IL" vs "Illinois". Consider a
   table mapping full state names to the short code.
3. **Domain-style names.** "Team Air Pvt. Ltd." is matched by "TEAMAIR.COM"
   and "teamair.com". Suggestion: remove a trailing `.com/.in/.net/.org/.co.in/.fr`
   and a leading `www.`. Today clean_text turns "teamair.com" into
   "teamair com".
4. **Junk tokens.** "N/A" ("CALUMET CITY, 351 HOXIE AVE, N/A, IL"), "##"
   ("##19821 WHEELWRIGHT DR"), a leading "--" ("-- Holloway Peak Inc
   Seafood"), "351- Hoxie Avenue". Most of this disappears in clean_text,
   but "n a" survives as two words. Drop an `n/a` / `na` token when it
   stands alone.
5. **Bracketed words.** "MS [Consultancy]", "Davis Family (Office)",
   "Prairie (Capital)", "Clairvoyant Ride Private (Limited)". Keep the word
   and drop the brackets (clean_text already does this). Make sure a
   bracketed legal form such as "(Limited)" is still recognised as a
   suffix.
6. **Doubled words.** "FH FH Business Pvt Ltd.", "Grand Grand Connecticut
   LLC". Collapse a word repeated immediately after itself.
7. **Hyphenated legal forms and reordering.** "Pvt-Ltd.", "PRAIRIE CAPITAL
   PLLC-PARTNERS". Also add **pllc** to Table 5.3 (it appears in US data).
8. **Typos and OCR-like swaps.** "M5 Consultancy" for "MS Consultancy",
   "Offie", "Rceoad" / "Rcdbr" for "Record". Normalization cannot fix these;
   the fuzzy features handle them. Just do not make them worse.
9. **House-number formats.** "Flat No:101" vs "Flat No:1-101", "7-1ST
   FLOOR", "120. Autumn Woods". extract_numbers should give the same
   useful numbers on both sides.
10. **Empty addresses.** About 3% of S2/S3 records have no address, and
    names are never empty. Empty input must still give empty strings and
    `[]`.
11. **Some matches share only the address.** "Team Air Pvt. Ltd." also
    matches a record named "Mirasol" at the same flat. Normalization cannot
    fix this. It is only a reason to keep `addr_core` clean.
