"""MA-owned interface checks (guide Section 5, v1.1). Must pass for the stubs AND for AM's/RA's real modules."""
import random
import time

import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, pair_features, pair_features_frame
from src.normalize import normalize_record

EXPECTED_FEATURES = [
    "name_ratio", "name_token_sort", "name_token_set", "core_jw", "core_token_set",
    "core_partial", "core_exact", "core_word_jaccard", "core_char3_jaccard",
    "first_word_match", "acronym_match", "legal_suffix_match", "dba_best",
    "name_len_ratio", "addr_token_set", "addr_token_sort", "addr_partial",
    "addr_word_jaccard", "postcode_match", "postcode_prefix3", "num_jaccard",
    "first_num_match", "landmark_sim", "addr_len_ratio", "country_match",
    "name_in_addr", "missing_count",
]
NORM_KEYS = ["entity_id", "country_norm", "name_norm", "name_core", "legal_suffix", "dba_name",
             "addr_norm", "addr_core", "postcode", "numbers", "landmark"]

_NAMES = ["Sharma & Sons Pvt. Ltd.", "ACME Corporation d/b/a Acme Tools", "SARL Boulangerie Dupont",
          "FH FH Business Pvt-Ltd.", "teamair.com", "MS [Consultancy]", "Davis Family Offie",
          "Clairvoyant Récord Private Ltd", "मध्य प्रदेश Traders", "-- Holloway Peak Inc Seafood", "", None]
_ADDRS = ["No. 12, 4th Cross, Near SBI ATM, Jayanagar, Bengaluru 560041",
          "1200 N. Main St., Suite 4, Springfield, IL 62704-1234",
          "12 bis Av. des Champs-Élysées, 75008 Paris", "##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD",
          "CALUMET CITY, 351 HOXIE AVE, N/A, IL", "602, Jaipur, RJ", "", float("nan")]
_COUNTRIES = ["India", "US", "France", "Germany", "", None]


def _records(n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    return [normalize_record({"entity_id": f"S2-{i}", "business_name": rnd.choice(_NAMES),
                              "business_address": rnd.choice(_ADDRS), "country": rnd.choice(_COUNTRIES)})
            for i in range(n)]


def test_names_and_keys():
    assert FEATURE_NAMES[:27] == EXPECTED_FEATURES  # extras may only be appended
    assert list(_records(1, 0)[0]) == NORM_KEYS


def test_frame_matches_reference_on_1000_pairs():
    a, b = _records(1000, 1), _records(1000, 2)
    got = pair_features_frame(pd.DataFrame(a), pd.DataFrame(b))
    assert list(got.columns) == FEATURE_NAMES
    assert all(got[c].dtype == np.float32 for c in got.columns)
    want = pd.DataFrame([pair_features(x, y) for x, y in zip(a, b)])[FEATURE_NAMES]
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), atol=1e-5)
    assert not got.isna().any().any()


def test_frame_speed_1m_pairs_per_minute():
    a, b = pd.DataFrame(_records(2000, 3) * 50), pd.DataFrame(_records(2000, 4) * 50)  # 100k pairs
    t = time.perf_counter()
    pair_features_frame(a, b)
    assert time.perf_counter() - t < 6.0  # 100k pairs in 6 s = 1M pairs/min
