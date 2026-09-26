import pytest
from src.normalize import normalize_record
from src.features import pair_features, FEATURE_NAMES

def test_f_beta_calculation():
    # Verify F_0.5 weighting: Precision is 2x more important than Recall
    beta = 0.5
    p, r = 0.9, 0.5
    f_05 = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
    assert round(f_05, 4) == 0.7759

def test_normalize_record():
    raw = {
        "entity_id": "TEST-1",
        "business_name": "Young, Sanchez & Downs Cambridge Inc.",
        "business_address": "2225 Sherwood Road, Kingsport, TN",
        "country": "us"
    }
    clean = normalize_record(raw)
    assert clean["country"] == "US"
    assert clean["country_norm"] == "us"
    assert clean["legal_suffix"] == "inc"
    assert "young" in clean["name_core"]

def test_pair_features():
    rec1 = normalize_record({"business_name": "walmart supercenter", "business_address": "100 main st", "country": "US"})
    rec2 = normalize_record({"business_name": "walmart", "business_address": "100 main street", "country": "US"})
    feats = pair_features(rec1, rec2)

    assert isinstance(feats, dict)
    assert "name_ratio" in feats
    assert "name_token_sort" in feats
    assert feats["country_match"] == 1.0
    assert feats["name_token_set"] > 0.5
    assert len(feats) == 27
    for v in feats.values():
        assert isinstance(v, (int, float))
        assert not (v != v)  # Check for NaN
