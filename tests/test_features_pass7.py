"""The five features MA appended in Pass 7: values on hand-built pairs, symmetry, and frame parity."""
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, pair_features, pair_features_frame
from src.normalize import normalize_record


def nr(name, addr="", country="US"):
    return normalize_record({"entity_id": "S2-1", "business_name": name, "business_address": addr, "country": country})


def feats(a, b):
    f = pair_features(a, b)
    assert f == pair_features(b, a)  # symmetric
    frame = pair_features_frame(pd.DataFrame([a]), pd.DataFrame([b])).iloc[0]
    np.testing.assert_allclose(frame[FEATURE_NAMES].to_numpy(), [f[k] for k in FEATURE_NAMES], atol=1e-6)
    return f


def test_house_number_conflict():
    assert feats(nr("Acme", "12 Main St, Springfield"), nr("Acme", "14 Main St, Springfield"))["house_number_conflict"] == 1.0
    assert feats(nr("Acme", "12 Main St"), nr("Acme", "012 Main Street"))["house_number_conflict"] == 0.0
    assert feats(nr("Acme", "Main St"), nr("Acme", "14 Main St"))["house_number_conflict"] == -1.0
    # '4th' is not a standalone house number; the postcode is excluded
    assert feats(nr("A", "4th Cross 560041", "India"), nr("A", "9 Cross 560041", "India"))["house_number_conflict"] == -1.0


def test_disjoint_brand_words():
    a, b = nr("Sharma Traders Pvt Ltd", "MG Rd, Pune"), nr("Sharma Pune Traders Limited", "MG Road, Pune")
    assert feats(a, b)["disjoint_brand_words"] == 0.0  # 'pune' is an address word, legal forms ignored
    assert feats(nr("Sharma Traders", "Pune"), nr("Verma Traders", "Pune"))["disjoint_brand_words"] == 2.0
    assert feats(nr(""), nr("Acme"))["disjoint_brand_words"] == -1.0


def test_branch_modifier_asymmetry():
    assert feats(nr("Acme Plumbing North"), nr("Acme Plumbing"))["branch_modifier_asymmetry"] == 1.0
    assert feats(nr("Teamsters Local 705"), nr("Teamsters Local 710"))["branch_modifier_asymmetry"] == 1.0
    assert feats(nr("Riverside Cafe"), nr("Riverside Cafe LLC"))["branch_modifier_asymmetry"] == 0.0
    assert feats(nr("Acme"), nr(""))["branch_modifier_asymmetry"] == -1.0


def test_name_exact_missing_addr():
    assert feats(nr("Acme Tools", ""), nr("Acme Tools", "1 Main St"))["name_exact_missing_addr"] == 1.0
    assert feats(nr("Acme Tools", "1 Main St"), nr("Acme Tools", "1 Main St"))["name_exact_missing_addr"] == 0.0
    assert feats(nr("Acme Tools", ""), nr("Zeta Foods", ""))["name_exact_missing_addr"] == 0.0


def test_core_nospace_ratio_catches_domain_style_names():
    f = feats(nr("urbangrill.com"), nr("Urban Grill!"))
    assert f["core_nospace_ratio"] == 1.0 and f["core_jw"] < 1.0
    assert feats(nr(""), nr("Urban Grill"))["core_nospace_ratio"] == -1.0
