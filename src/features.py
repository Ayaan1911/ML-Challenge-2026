# STUB - replaced by teammate branch; do not extend here.
"""src/features.py - owner RA. Imports allowed: stdlib, numpy, rapidfuzz."""
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES: list[str] = [
    "name_ratio", "name_token_sort", "name_token_set", "core_jw", "core_token_set",
    "core_partial", "core_exact", "core_word_jaccard", "core_char3_jaccard",
    "first_word_match", "acronym_match", "legal_suffix_match", "dba_best",
    "name_len_ratio", "addr_token_set", "addr_token_sort", "addr_partial",
    "addr_word_jaccard", "postcode_match", "postcode_prefix3", "num_jaccard",
    "first_num_match", "landmark_sim", "addr_len_ratio", "country_match",
    "name_in_addr", "missing_count",
]


def _sim(fn, x: str, y: str) -> float:
    return fn(x, y) / 100.0 if x and y else -1.0


def _eq(x, y) -> float:
    return -1.0 if not x or not y else float(x == y)


def pair_features(a: dict, b: dict) -> dict[str, float]:
    g = lambda r, k: r.get(k) or ""  # noqa: E731
    f = dict.fromkeys(FEATURE_NAMES, -1.0)
    an, bn = g(a, "name_norm"), g(b, "name_norm")
    ac, bc = g(a, "name_core"), g(b, "name_core")
    aa, ba = g(a, "addr_core"), g(b, "addr_core")
    f["name_ratio"] = _sim(fuzz.ratio, an, bn)
    f["name_token_sort"] = _sim(fuzz.token_sort_ratio, an, bn)
    f["name_token_set"] = _sim(fuzz.token_set_ratio, an, bn)
    f["core_jw"] = float(JaroWinkler.similarity(ac, bc)) if ac and bc else -1.0
    f["core_token_set"] = _sim(fuzz.token_set_ratio, ac, bc)
    f["core_exact"] = _eq(ac, bc)
    f["addr_token_set"] = _sim(fuzz.token_set_ratio, g(a, "addr_norm"), g(b, "addr_norm"))
    f["addr_token_sort"] = _sim(fuzz.token_sort_ratio, aa, ba)
    f["postcode_match"] = _eq(g(a, "postcode"), g(b, "postcode"))
    f["country_match"] = _eq(g(a, "country_norm"), g(b, "country_norm"))
    f["missing_count"] = float(sum(not g(r, k) for r in (a, b) for k in ("name_core", "addr_core", "postcode")))
    return f


def pair_features_batch(a_list: list[dict], b_list: list[dict]) -> list[dict[str, float]]:
    return [pair_features(a, b) for a, b in zip(a_list, b_list)]
