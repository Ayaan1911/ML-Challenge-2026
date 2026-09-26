"""src/features.py - owner RA. Imports allowed: stdlib, numpy, rapidfuzz."""

from typing import Any
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES: list[str] = [
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


def _get_str(record: dict[str, Any], key: str) -> str:
    """Safely extract a stripped string value from record dict."""
    val = record.get(key)
    if val is None or not isinstance(val, str):
        return ""
    return val.strip()


def _get_list(record: dict[str, Any], key: str) -> list[str]:
    """Safely extract a list of strings from record dict."""
    val = record.get(key)
    if not isinstance(val, list):
        return []
    return [str(x).strip() for x in val if x is not None and str(x).strip() != ""]


def pair_features(a: dict, b: dict) -> dict[str, float]:
    """a = NormRecord of the Source 1 record, b = NormRecord of a candidate
    (Source 2 or 3). Returns a dict with EVERY key in FEATURE_NAMES, all Python
    floats. Similarities are in [0, 1] (divide rapidfuzz scores by 100).
    If a feature cannot be computed because a needed field is empty on either
    side, return -1.0 for it (never 0, never NaN). Must be symmetric:
    pair_features(a, b) == pair_features(b, a).
    """
    name_norm_a = _get_str(a, "name_norm")
    name_norm_b = _get_str(b, "name_norm")
    core_a = _get_str(a, "name_core")
    core_b = _get_str(b, "name_core")
    suf_a = _get_str(a, "legal_suffix")
    suf_b = _get_str(b, "legal_suffix")
    dba_a = _get_str(a, "dba_name")
    dba_b = _get_str(b, "dba_name")
    addr_norm_a = _get_str(a, "addr_norm")
    addr_norm_b = _get_str(b, "addr_norm")
    addr_core_a = _get_str(a, "addr_core")
    addr_core_b = _get_str(b, "addr_core")
    pc_a = _get_str(a, "postcode")
    pc_b = _get_str(b, "postcode")
    nums_a = _get_list(a, "numbers")
    nums_b = _get_list(b, "numbers")
    lm_a = _get_str(a, "landmark")
    lm_b = _get_str(b, "landmark")
    c_a = _get_str(a, "country_norm")
    c_b = _get_str(b, "country_norm")

    feats: dict[str, float] = {}

    # 1. name_ratio: fuzz.ratio(name_norm) / 100
    if not name_norm_a or not name_norm_b:
        feats["name_ratio"] = -1.0
    else:
        feats["name_ratio"] = float(fuzz.ratio(name_norm_a, name_norm_b)) / 100.0

    # 2. name_token_sort: fuzz.token_sort_ratio(name_norm) / 100
    if not name_norm_a or not name_norm_b:
        feats["name_token_sort"] = -1.0
    else:
        feats["name_token_sort"] = float(fuzz.token_sort_ratio(name_norm_a, name_norm_b)) / 100.0

    # 3. name_token_set: fuzz.token_set_ratio(name_norm) / 100
    if not name_norm_a or not name_norm_b:
        feats["name_token_set"] = -1.0
    else:
        feats["name_token_set"] = float(fuzz.token_set_ratio(name_norm_a, name_norm_b)) / 100.0

    # 4. core_jw: JaroWinkler.similarity(name_core)
    if not core_a or not core_b:
        feats["core_jw"] = -1.0
    else:
        feats["core_jw"] = float(JaroWinkler.similarity(core_a, core_b))

    # 5. core_token_set: fuzz.token_set_ratio(name_core) / 100
    if not core_a or not core_b:
        feats["core_token_set"] = -1.0
    else:
        feats["core_token_set"] = float(fuzz.token_set_ratio(core_a, core_b)) / 100.0

    # 6. core_partial: fuzz.partial_ratio(name_core) / 100
    if not core_a or not core_b:
        feats["core_partial"] = -1.0
    else:
        feats["core_partial"] = float(fuzz.partial_ratio(core_a, core_b)) / 100.0

    # 7. core_exact: 1.0 if name_core equal, else 0.0
    if not core_a or not core_b:
        feats["core_exact"] = -1.0
    else:
        feats["core_exact"] = 1.0 if core_a == core_b else 0.0

    # Split word tokens once for name_core if present
    words_core_a = core_a.split() if core_a else []
    words_core_b = core_b.split() if core_b else []

    # 8. core_word_jaccard: Jaccard of the word sets of name_core
    if not core_a or not core_b:
        feats["core_word_jaccard"] = -1.0
    else:
        set_wa, set_wb = set(words_core_a), set(words_core_b)
        if not set_wa or not set_wb:
            feats["core_word_jaccard"] = -1.0
        else:
            union_w = set_wa | set_wb
            feats["core_word_jaccard"] = float(len(set_wa & set_wb)) / float(len(union_w))

    # 9. core_char3_jaccard: Jaccard of character 3-gram sets of name_core with spaces removed
    if not core_a or not core_b:
        feats["core_char3_jaccard"] = -1.0
    else:
        ns_a = core_a.replace(" ", "")
        ns_b = core_b.replace(" ", "")
        if len(ns_a) < 3 or len(ns_b) < 3:
            feats["core_char3_jaccard"] = -1.0
        else:
            g_a = {ns_a[i : i + 3] for i in range(len(ns_a) - 2)}
            g_b = {ns_b[i : i + 3] for i in range(len(ns_b) - 2)}
            if not g_a or not g_b:
                feats["core_char3_jaccard"] = -1.0
            else:
                union_g = g_a | g_b
                feats["core_char3_jaccard"] = float(len(g_a & g_b)) / float(len(union_g))

    # 10. first_word_match: 1.0 if first word of name_core equal, else 0.0
    if not core_a or not core_b:
        feats["first_word_match"] = -1.0
    else:
        if not words_core_a or not words_core_b:
            feats["first_word_match"] = -1.0
        else:
            feats["first_word_match"] = 1.0 if words_core_a[0] == words_core_b[0] else 0.0

    # 11. acronym_match: 1.0 if acronym (first letters of words) of one core, length >= 2,
    # equals the other core with spaces removed; or both acronyms equal with length >= 3; else 0.0
    if not core_a or not core_b:
        feats["acronym_match"] = -1.0
    else:
        acr_a = "".join(w[0] for w in words_core_a if w)
        acr_b = "".join(w[0] for w in words_core_b if w)
        ns_a = core_a.replace(" ", "")
        ns_b = core_b.replace(" ", "")
        is_match = (
            (len(acr_a) >= 2 and acr_a == ns_b)
            or (len(acr_b) >= 2 and acr_b == ns_a)
            or (len(acr_a) >= 3 and len(acr_b) >= 3 and acr_a == acr_b)
        )
        feats["acronym_match"] = 1.0 if is_match else 0.0

    # 12. legal_suffix_match: 1.0 equal, 0.0 different, -1.0 if either empty
    if not suf_a or not suf_b:
        feats["legal_suffix_match"] = -1.0
    else:
        feats["legal_suffix_match"] = 1.0 if suf_a == suf_b else 0.0

    # 13. dba_best: max fuzz.token_set_ratio / 100 over all pairs from {name_core, dba_name} of a and of b, ignoring empty strings
    cand_a = [s for s in (core_a, dba_a) if s]
    cand_b = [s for s in (core_b, dba_b) if s]
    if not cand_a or not cand_b:
        feats["dba_best"] = -1.0
    else:
        feats["dba_best"] = max(
            float(fuzz.token_set_ratio(sa, sb)) / 100.0 for sa in cand_a for sb in cand_b
        )

    # 14. name_len_ratio: min(len) / max(len) of name_core
    if not core_a or not core_b:
        feats["name_len_ratio"] = -1.0
    else:
        len_a, len_b = len(core_a), len(core_b)
        feats["name_len_ratio"] = float(min(len_a, len_b)) / float(max(len_a, len_b))

    # 15. addr_token_set: fuzz.token_set_ratio(addr_norm) / 100
    if not addr_norm_a or not addr_norm_b:
        feats["addr_token_set"] = -1.0
    else:
        feats["addr_token_set"] = float(fuzz.token_set_ratio(addr_norm_a, addr_norm_b)) / 100.0

    # 16. addr_token_sort: fuzz.token_sort_ratio(addr_core) / 100
    if not addr_core_a or not addr_core_b:
        feats["addr_token_sort"] = -1.0
    else:
        feats["addr_token_sort"] = float(fuzz.token_sort_ratio(addr_core_a, addr_core_b)) / 100.0

    # 17. addr_partial: fuzz.partial_ratio(addr_core) / 100
    if not addr_core_a or not addr_core_b:
        feats["addr_partial"] = -1.0
    else:
        feats["addr_partial"] = float(fuzz.partial_ratio(addr_core_a, addr_core_b)) / 100.0

    # 18. addr_word_jaccard: Jaccard of the word sets of addr_core
    if not addr_core_a or not addr_core_b:
        feats["addr_word_jaccard"] = -1.0
    else:
        words_addr_a = set(addr_core_a.split())
        words_addr_b = set(addr_core_b.split())
        if not words_addr_a or not words_addr_b:
            feats["addr_word_jaccard"] = -1.0
        else:
            union_aw = words_addr_a | words_addr_b
            feats["addr_word_jaccard"] = float(len(words_addr_a & words_addr_b)) / float(
                len(union_aw)
            )

    # 19. postcode_match: 1.0 equal, 0.0 different, -1.0 if either empty
    if not pc_a or not pc_b:
        feats["postcode_match"] = -1.0
    else:
        feats["postcode_match"] = 1.0 if pc_a == pc_b else 0.0

    # 20. postcode_prefix3: 1.0 if first 3 digits equal, 0.0 if not, -1.0 if either empty
    if not pc_a or not pc_b or len(pc_a) < 3 or len(pc_b) < 3:
        feats["postcode_prefix3"] = -1.0
    else:
        feats["postcode_prefix3"] = 1.0 if pc_a[:3] == pc_b[:3] else 0.0

    # 21. num_jaccard: Jaccard of the numbers lists as sets; -1.0 if either empty
    if not nums_a or not nums_b:
        feats["num_jaccard"] = -1.0
    else:
        set_nums_a = set(nums_a)
        set_nums_b = set(nums_b)
        if not set_nums_a or not set_nums_b:
            feats["num_jaccard"] = -1.0
        else:
            union_nums = set_nums_a | set_nums_b
            feats["num_jaccard"] = float(len(set_nums_a & set_nums_b)) / float(len(union_nums))

    # 22. first_num_match: 1.0 if numbers[0] equal, 0.0 if not, -1.0 if either empty
    if not nums_a or not nums_b:
        feats["first_num_match"] = -1.0
    else:
        feats["first_num_match"] = 1.0 if nums_a[0] == nums_b[0] else 0.0

    # 23. landmark_sim: fuzz.token_set_ratio(landmark) / 100; -1.0 if either empty
    if not lm_a or not lm_b:
        feats["landmark_sim"] = -1.0
    else:
        feats["landmark_sim"] = float(fuzz.token_set_ratio(lm_a, lm_b)) / 100.0

    # 24. addr_len_ratio: min(len) / max(len) of addr_core
    if not addr_core_a or not addr_core_b:
        feats["addr_len_ratio"] = -1.0
    else:
        len_addr_a, len_addr_b = len(addr_core_a), len(addr_core_b)
        feats["addr_len_ratio"] = float(min(len_addr_a, len_addr_b)) / float(
            max(len_addr_a, len_addr_b)
        )

    # 25. country_match: 1.0 equal, 0.0 different, -1.0 if either empty
    if not c_a or not c_b:
        feats["country_match"] = -1.0
    else:
        feats["country_match"] = 1.0 if c_a == c_b else 0.0

    # 26. name_in_addr: 1.0 if name_core of either record (length >= 4) appears inside the other's addr_norm as whole words, else 0.0
    if not core_a or not core_b or not addr_norm_a or not addr_norm_b:
        feats["name_in_addr"] = -1.0
    else:
        in_b = len(core_a) >= 4 and f" {core_a} " in f" {addr_norm_b} "
        in_a = len(core_b) >= 4 and f" {core_b} " in f" {addr_norm_a} "
        feats["name_in_addr"] = 1.0 if (in_b or in_a) else 0.0

    # 27. missing_count: number of empty fields among name_core, addr_core, postcode on both sides combined (0 to 6), as float
    empty_count = 0
    for r in (a, b):
        if not _get_str(r, "name_core"):
            empty_count += 1
        if not _get_str(r, "addr_core"):
            empty_count += 1
        if not _get_str(r, "postcode"):
            empty_count += 1
    feats["missing_count"] = float(empty_count)

    return feats


def pair_features_batch(a_list: list[dict], b_list: list[dict]) -> list[dict[str, float]]:
    """Same as calling pair_features on zip(a_list, b_list). Simple loop is fine."""
    return [pair_features(a, b) for a, b in zip(a_list, b_list)]


# ---- MA addition (interface v1.1): vectorized pair_features over row-aligned NormRecord frames ----
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rapidfuzz import process  # noqa: E402


def _fcol(df: pd.DataFrame, k: str) -> list[str]:
    if k not in df:
        return [""] * len(df)
    return [v.strip() if isinstance(v, str) else "" for v in df[k].tolist()]


def _flist(df: pd.DataFrame, k: str) -> list[list[str]]:
    """Like _get_list, but also accepts the numpy arrays Arrow produces for list columns."""
    if k not in df:
        return [[] for _ in range(len(df))]
    out = []
    for v in df[k].tolist():
        seq = v.tolist() if isinstance(v, np.ndarray) else v if isinstance(v, (list, tuple)) else []
        out.append([str(x).strip() for x in seq if x is not None and str(x).strip() != ""])
    return out


def _fsim(scorer, x: list[str], y: list[str], ok: np.ndarray | None = None, scale: float = 100.0) -> np.ndarray:
    xa, ya = np.array(x, dtype=object), np.array(y, dtype=object)
    ok = (xa != "") & (ya != "") if ok is None else ok
    out = np.full(len(xa), -1.0, dtype=np.float64)
    if ok.any():
        out[ok] = process.cpdist(xa[ok].tolist(), ya[ok].tolist(), scorer=scorer, workers=-1,
                                 dtype=np.float64) / scale
    return out


def _feq(x: list[str], y: list[str], n: int = 0) -> np.ndarray:
    """1.0 equal, 0.0 different, -1.0 if either empty (or shorter than n when comparing prefixes)."""
    return np.array([-1.0 if not a or not b or len(a) < max(n, 1) or len(b) < max(n, 1)
                     else float((a[:n] == b[:n]) if n else (a == b)) for a, b in zip(x, y)])


def _jac(sa: set, sb: set) -> float:
    return len(sa & sb) / len(sa | sb) if sa and sb else -1.0


def _g3(s: str) -> set:
    s = s.replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _acronym(ca: str, cb: str) -> float:
    wa, wb = ca.split(), cb.split()
    aa, ab = "".join(w[0] for w in wa if w), "".join(w[0] for w in wb if w)
    na, nb = ca.replace(" ", ""), cb.replace(" ", "")
    return float((len(aa) >= 2 and aa == nb) or (len(ab) >= 2 and ab == na)
                 or (len(aa) >= 3 and len(ab) >= 3 and aa == ab))


def _name_in_addr(ca: str, cb: str, aa: str, ab: str) -> float:
    if not ca or not cb or not aa or not ab:
        return -1.0
    return float((len(ca) >= 4 and f" {ca} " in f" {ab} ") or (len(cb) >= 4 and f" {cb} " in f" {aa} "))


def pair_features_frame(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Vectorized pair_features over row-aligned NormRecord frames: float32 columns in FEATURE_NAMES order,
    equal to pair_features row by row (checked in tests/test_interface.py)."""
    na, nb = _fcol(a, "name_norm"), _fcol(b, "name_norm")
    ca, cb = _fcol(a, "name_core"), _fcol(b, "name_core")
    da, db = _fcol(a, "dba_name"), _fcol(b, "dba_name")
    aa, ab = _fcol(a, "addr_norm"), _fcol(b, "addr_norm")
    ra, rb = _fcol(a, "addr_core"), _fcol(b, "addr_core")
    pa_, pb = _fcol(a, "postcode"), _fcol(b, "postcode")
    ma, mb = _flist(a, "numbers"), _flist(b, "numbers")
    core_ok = np.array([bool(x and y) for x, y in zip(ca, cb)])
    addr_ok = np.array([bool(x and y) for x, y in zip(ra, rb)])
    neg = lambda ok, vals: np.where(ok, vals, -1.0)  # noqa: E731
    f = {
        "name_ratio": _fsim(fuzz.ratio, na, nb),
        "name_token_sort": _fsim(fuzz.token_sort_ratio, na, nb),
        "name_token_set": _fsim(fuzz.token_set_ratio, na, nb),
        "core_jw": _fsim(JaroWinkler.similarity, ca, cb, scale=1.0),
        "core_token_set": _fsim(fuzz.token_set_ratio, ca, cb),
        "core_partial": _fsim(fuzz.partial_ratio, ca, cb),
        "core_exact": _feq(ca, cb),
        "core_word_jaccard": np.array([_jac(set(x.split()), set(y.split())) if x and y else -1.0 for x, y in zip(ca, cb)]),
        "core_char3_jaccard": np.array([_jac(_g3(x), _g3(y)) if x and y else -1.0 for x, y in zip(ca, cb)]),
        "first_word_match": np.array([float(x.split()[0] == y.split()[0]) if x.split() and y.split() else -1.0
                                      for x, y in zip(ca, cb)]),
        "acronym_match": neg(core_ok, [_acronym(x, y) if x and y else -1.0 for x, y in zip(ca, cb)]),
        "legal_suffix_match": _feq(_fcol(a, "legal_suffix"), _fcol(b, "legal_suffix")),
        # dba_best: max token_set_ratio over non-empty {name_core, dba_name} x {name_core, dba_name}
        "dba_best": np.maximum.reduce([_fsim(fuzz.token_set_ratio, x, y) for x, y in ((ca, cb), (ca, db), (da, cb), (da, db))]),
        "name_len_ratio": neg(core_ok, [min(len(x), len(y)) / max(len(x), len(y)) if x and y else -1.0 for x, y in zip(ca, cb)]),
        "addr_token_set": _fsim(fuzz.token_set_ratio, aa, ab),
        "addr_token_sort": _fsim(fuzz.token_sort_ratio, ra, rb),
        "addr_partial": _fsim(fuzz.partial_ratio, ra, rb),
        "addr_word_jaccard": np.array([_jac(set(x.split()), set(y.split())) if x and y else -1.0 for x, y in zip(ra, rb)]),
        "postcode_match": _feq(pa_, pb),
        "postcode_prefix3": _feq(pa_, pb, 3),
        "num_jaccard": np.array([_jac(set(x), set(y)) if x and y else -1.0 for x, y in zip(ma, mb)]),
        "first_num_match": np.array([float(x[0] == y[0]) if x and y else -1.0 for x, y in zip(ma, mb)]),
        "landmark_sim": _fsim(fuzz.token_set_ratio, _fcol(a, "landmark"), _fcol(b, "landmark")),
        "addr_len_ratio": neg(addr_ok, [min(len(x), len(y)) / max(len(x), len(y)) if x and y else -1.0 for x, y in zip(ra, rb)]),
        "country_match": _feq(_fcol(a, "country_norm"), _fcol(b, "country_norm")),
        "name_in_addr": np.array([_name_in_addr(w, x, y, z) for w, x, y, z in zip(ca, cb, aa, ab)]),
        "missing_count": sum(np.array([not v for v in _fcol(r, k)], dtype=np.float64)
                             for r in (a, b) for k in ("name_core", "addr_core", "postcode")),
    }
    return pd.DataFrame({k: np.asarray(f[k], dtype=np.float32) for k in FEATURE_NAMES}, index=a.index)
