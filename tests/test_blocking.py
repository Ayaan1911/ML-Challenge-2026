import numpy as np

from src.blocking import exact_pairs


def test_exact_pairs_join_cap_and_empty():
    s1 = np.array(["a", "b", "a", "", "c", "c", "c"], dtype=object)
    rec = np.array(["c", "a", "", "z", "b"], dtype=object)
    r, s = exact_pairs(s1, rec, cap=2)  # "c" is held by 3 S1 -> skipped; "" never joins
    assert sorted(zip(r.tolist(), s.tolist())) == [(1, 0), (1, 2), (4, 1)]
    assert list(r) == sorted(r)  # sorted by record index, blocking slices it with searchsorted


def test_tfidf_prefix_topk_matches_full_cosine():
    from joblib import Parallel

    from src.blocking import CONFIG, TfidfIndex
    s1 = ["sharma sons", "acme tools", "boulangerie dupont", "team air", "davis family office", "grand connecticut"]
    with Parallel(n_jobs=1) as par:
        idx = TfidfIndex(s1, dict(CONFIG, threads=1), par, prefix=2)
        q = idx.vectorize(["sharma and sons", "teamair", "xyz", "davis family offie"])
        r, s, c = idx.topk(q, 1)
        full = (q @ idx.a.T).toarray()
    assert list(zip(r.tolist(), s.tolist())) == [(0, 0), (1, 3), (3, 4)]  # "xyz" has no hit above min_cos
    np.testing.assert_allclose(c, full[r, s], rtol=1e-5)


def test_prune_mask_keeps_keys_and_top_n_above_floor():
    from src.blocking import prune_mask
    rec = np.array([0, 0, 0, 0, 1, 1])
    has_key = np.array([False, True, False, False, False, False])
    score = np.array([0.9, 0.1, 0.8, 0.2, 0.5, 0.35])
    # rec 0: key hit always kept; top-2 by score are 0.9 and 0.8; rec 1: 0.35 is below the 0.4 floor
    assert prune_mask(rec, has_key, score, 2, 0.4).tolist() == [True, True, True, False, True, False]
    assert prune_mask(rec, has_key, score, None, 0.4).all()


def test_word_postcode_keys_skip_short_words_and_postcode():
    import pandas as pd

    from src.blocking import word_postcode_keys
    df = pd.DataFrame({"addr_core": ["no 12 anuska towers 560041", "main rd", ""],
                       "postcode": ["560041", "", "110001"]})
    keys, rows = word_postcode_keys(df)
    assert sorted(zip(keys.tolist(), rows.tolist())) == [
        ("12|560041", 0), ("anuska|560041", 0), ("towers|560041", 0)]


def test_prune_ranks_rare_word_hits_above_weak_cosine_lookalikes():
    import pandas as pd

    from src.blocking import ALL_KEYS, prune
    df = pd.DataFrame({"s1_id": ["a", "b", "c", "d"], "cand_id": ["r"] * 4,
                       "tfidf_name_cos": [0.6, 0.55, 0.0, 0.0], "tfidf_nameaddr_cos": [0.2, 0.2, 0.35, 0.0],
                       "addr_word_idf": [0.0, 0.0, 30.0, 0.0]})
    for k in ALL_KEYS:
        df[k] = 0
    df.loc[3, "key_name_nospace"] = 1  # a key hit is always kept
    kept = prune(df, n=1)["s1_id"].tolist()
    assert kept == ["c", "d"]  # rare shared address words (idf 30) outrank a 0.6 name look-alike
