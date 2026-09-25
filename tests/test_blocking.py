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
