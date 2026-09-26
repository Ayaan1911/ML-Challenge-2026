import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import compute_f_beta
from src.normalize import normalize_record
from src.features import pair_features

def test_f_beta_calculation():
    # In F0.5, precision is weighted 2x over recall
    # If P=1.0 and R=0.5: F0.5 = 1.25 * (1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    f05 = compute_f_beta(precision=1.0, recall=0.5, beta=0.5)
    assert round(f05, 4) == 0.8333

def test_normalize_record():
    raw = {
        "entity_id": "TEST-1",
        "business_name": "Young, Sanchez & Downs Cambridge Inc.",
        "business_address": "2225 Sherwood Road, Kingsport, TN",
        "country": "us"
    }
    clean = normalize_record(raw)
    assert clean["country"] == "US"
    assert "inc" not in clean["business_name"].lower().split()
    assert clean["entity_id"] == "TEST-1"

def test_pair_features():
    rec1 = {"business_name": "walmart supercenter", "business_address": "100 main st", "country": "US"}
    rec2 = {"business_name": "walmart", "business_address": "100 main street", "country": "US"}
    feats = pair_features(rec1, rec2)

    assert isinstance(feats, dict)
    assert "name_jaccard" in feats
    assert feats["country_match"] == 1.0
    for v in feats.values():
        assert isinstance(v, (int, float))
        assert not (v != v)  # Check for NaN
