"""tests/test_features.py - owner RA. Unit tests for src/features.py using pytest."""

import os
import sys
import time
import pytest

# Ensure repository root is on sys.path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FEATURE_NAMES, pair_features, pair_features_batch

# Exact NormRecord test fixtures from Section 5.4 of TEAM_GUIDE.md

OUT_1 = {
    "entity_id": "S1-00001",
    "country_norm": "india",
    "name_norm": "sharma and sons pvt ltd",
    "name_core": "sharma and sons",
    "legal_suffix": "pvt ltd",
    "dba_name": "",
    "addr_norm": "no 12 4th cross near sbi atm jayanagar bangalore 560041",
    "addr_core": "no 12 4th cross jayanagar bangalore",
    "postcode": "560041",
    "numbers": ["12", "4"],
    "landmark": "sbi atm",
}

OUT_2 = {
    "entity_id": "S2-00047",
    "country_norm": "us",
    "name_norm": "acme corp dba acme tools",
    "name_core": "acme",
    "legal_suffix": "corp",
    "dba_name": "acme tools",
    "addr_norm": "1200 n main st ste 4 springfield il 62704 1234",
    "addr_core": "1200 n main st ste 4 springfield il 1234",
    "postcode": "62704",
    "numbers": ["1200", "4", "1234"],
    "landmark": "",
}

OUT_3 = {
    "entity_id": "S3-00812",
    "country_norm": "france",
    "name_norm": "sarl boulangerie dupont",
    "name_core": "boulangerie dupont",
    "legal_suffix": "sarl",
    "dba_name": "",
    "addr_norm": "12 bis av des champs elysees 75008 paris",
    "addr_core": "12 bis av des champs elysees paris",
    "postcode": "75008",
    "numbers": ["12"],
    "landmark": "",
}

OUT_4 = {
    "entity_id": "S2-99999",
    "country_norm": "germany",
    "name_norm": "",
    "name_core": "",
    "legal_suffix": "",
    "dba_name": "",
    "addr_norm": "",
    "addr_core": "",
    "postcode": "",
    "numbers": [],
    "landmark": "",
}

ALL_EMPTY = {
    "entity_id": "",
    "country_norm": "",
    "name_norm": "",
    "name_core": "",
    "legal_suffix": "",
    "dba_name": "",
    "addr_norm": "",
    "addr_core": "",
    "postcode": "",
    "numbers": [],
    "landmark": "",
}


def test_feature_names_order_and_count():
    """Verify exactly 27 feature names matching Table 5.5."""
    assert len(FEATURE_NAMES) == 27
    expected_order = [
        "name_ratio",
        "name_token_sort",
        "name_token_set",
        "core_jw",
        "core_token_set",
        "core_partial",
        "core_exact",
        "core_word_jaccard",
        "core_char3_jaccard",
        "first_word_match",
        "acronym_match",
        "legal_suffix_match",
        "dba_best",
        "name_len_ratio",
        "addr_token_set",
        "addr_token_sort",
        "addr_partial",
        "addr_word_jaccard",
        "postcode_match",
        "postcode_prefix3",
        "num_jaccard",
        "first_num_match",
        "landmark_sim",
        "addr_len_ratio",
        "country_match",
        "name_in_addr",
        "missing_count",
    ]
    assert FEATURE_NAMES == expected_order


def test_keys_present_and_all_floats():
    """Verify every feature key is present and strictly a Python float."""
    pairs = [
        (OUT_1, OUT_1),
        (OUT_1, OUT_2),
        (OUT_1, OUT_3),
        (OUT_2, OUT_3),
        (OUT_1, OUT_4),
        (OUT_4, OUT_4),
        (ALL_EMPTY, ALL_EMPTY),
    ]
    for a, b in pairs:
        feats = pair_features(a, b)
        assert list(feats.keys()) == FEATURE_NAMES
        for k, v in feats.items():
            assert isinstance(v, float), f"Key {k} is not float: {type(v)}"


def test_value_ranges():
    """All values in [-1, 1] except missing_count in [0, 6]."""
    pairs = [
        (OUT_1, OUT_1),
        (OUT_1, OUT_2),
        (OUT_1, OUT_3),
        (OUT_2, OUT_4),
        (OUT_3, OUT_4),
        (OUT_4, OUT_4),
        (ALL_EMPTY, OUT_1),
    ]
    for a, b in pairs:
        feats = pair_features(a, b)
        for k, v in feats.items():
            if k == "missing_count":
                assert 0.0 <= v <= 6.0, f"missing_count out of range: {v}"
            else:
                assert -1.0 <= v <= 1.0, f"Feature {k} out of [-1, 1] range: {v}"


def test_symmetry():
    """Verify pair_features(a, b) == pair_features(b, a) across all test fixtures."""
    fixtures = [OUT_1, OUT_2, OUT_3, OUT_4, ALL_EMPTY]
    for i in range(len(fixtures)):
        for j in range(i, len(fixtures)):
            a, b = fixtures[i], fixtures[j]
            f_ab = pair_features(a, b)
            f_ba = pair_features(b, a)
            assert f_ab == f_ba, f"Symmetry violated between fixture {i} and {j}"


def test_out1_self_match():
    """OUT_1 versus itself gives 1.0 on features 1 to 6 and 15 to 17, and postcode_match 1.0."""
    res = pair_features(OUT_1, OUT_1)

    # Features 1 to 6
    for f in [
        "name_ratio",
        "name_token_sort",
        "name_token_set",
        "core_jw",
        "core_token_set",
        "core_partial",
    ]:
        assert res[f] == 1.0, f"{f} should be 1.0, got {res[f]}"

    # Features 15 to 17
    for f in ["addr_token_set", "addr_token_sort", "addr_partial"]:
        assert res[f] == 1.0, f"{f} should be 1.0, got {res[f]}"

    # Postcode match
    assert res["postcode_match"] == 1.0

    # Missing count for OUT_1 vs OUT_1 should be 0.0
    assert res["missing_count"] == 0.0


def test_country_match_out1_vs_out3():
    """OUT_1 (India) versus OUT_3 (France) gives country_match 0.0."""
    res = pair_features(OUT_1, OUT_3)
    assert res["country_match"] == 0.0


def test_all_empty_record():
    """An all-empty record gives -1.0 where specified and missing_count 6.0, no crash."""
    res = pair_features(ALL_EMPTY, ALL_EMPTY)
    assert res["missing_count"] == 6.0
    for k in FEATURE_NAMES:
        if k != "missing_count":
            assert res[k] == -1.0, f"Expected -1.0 for empty pair feature {k}, got {res[k]}"


def test_out4_empty_fields_handling():
    """OUT_4 has non-empty country ('germany') but empty name and address fields."""
    res = pair_features(OUT_4, OUT_4)
    # country matches
    assert res["country_match"] == 1.0
    # missing_count is 6.0
    assert res["missing_count"] == 6.0
    # name and address features return -1.0
    assert res["name_ratio"] == -1.0
    assert res["core_jw"] == -1.0
    assert res["addr_token_set"] == -1.0
    assert res["postcode_match"] == -1.0
    assert res["num_jaccard"] == -1.0
    assert res["landmark_sim"] == -1.0


def test_bad_and_missing_input_no_crash():
    """Inputs with None, NaN, missing keys, or unexpected types must never crash."""
    bad_rec_1 = {
        "entity_id": None,
        "name_norm": None,
        "name_core": None,
        "numbers": None,
        "postcode": None,
    }
    bad_rec_2 = {
        "business_name": float("nan"),
        "numbers": [12, None, "34"],
    }
    res = pair_features(bad_rec_1, bad_rec_2)
    assert list(res.keys()) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in res.values())
    assert pair_features({}, {}) == pair_features(ALL_EMPTY, ALL_EMPTY)


def test_acronym_match_logic():
    """Test acronym match rule (length >= 2 equals nospace core, or both length >= 3 match)."""
    rec_a = {"name_core": "state bank india"}
    rec_b = {"name_core": "sbi"}
    assert pair_features(rec_a, rec_b)["acronym_match"] == 1.0
    assert pair_features(rec_b, rec_a)["acronym_match"] == 1.0

    rec_c = {"name_core": "state bank industries"}
    assert pair_features(rec_a, rec_c)["acronym_match"] == 1.0

    rec_d = {"name_core": "state bank"}
    rec_e = {"name_core": "national bank"}
    assert pair_features(rec_d, rec_e)["acronym_match"] == 0.0


def test_name_in_addr_logic():
    """Test name_in_addr: 1.0 if core (len >= 4) appears as whole words in other's addr_norm."""
    rec_a = {
        "name_core": "acme",
        "addr_norm": "1200 high st",
    }
    rec_b = {
        "name_core": "tools corp",
        "addr_norm": "450 acme road suite 12",
    }
    # "acme" is in rec_b's addr_norm as whole words
    assert pair_features(rec_a, rec_b)["name_in_addr"] == 1.0
    assert pair_features(rec_b, rec_a)["name_in_addr"] == 1.0

    rec_c = {
        "name_core": "acme",
        "addr_norm": "450 acmed road suite 12",  # subword, not whole word
    }
    assert pair_features(rec_a, rec_c)["name_in_addr"] == 0.0


def test_pair_features_batch():
    """Test pair_features_batch equals elementwise pair_features."""
    a_list = [OUT_1, OUT_2, OUT_3]
    b_list = [OUT_2, OUT_3, OUT_1]
    batch_res = pair_features_batch(a_list, b_list)
    assert len(batch_res) == 3
    for idx, (a, b) in enumerate(zip(a_list, b_list)):
        assert batch_res[idx] == pair_features(a, b)


def test_speed_20000_calls():
    """Speed test: 20,000 calls to pair_features should finish in under about 10 seconds."""
    t0 = time.time()
    for _ in range(20000):
        pair_features(OUT_1, OUT_2)
    elapsed = time.time() - t0
    print(f"\n20,000 pair_features calls finished in {elapsed:.3f} seconds")
    assert elapsed < 10.0, f"Speed test failed: {elapsed:.2f}s >= 10.0s"
