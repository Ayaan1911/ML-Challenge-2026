import numpy as np
import pandas as pd

from src.data import id_key, key_id
from src.metric import macro_f05
from src.postprocess import Eval, best_per_record, decide, f05_per_s1
from src.run_pipeline import write_grouped


def test_id_key_round_trip():
    ids = ["S1-5", "S2-158121477", "S3-999999999"]
    assert key_id(id_key(ids)).to_pylist() == ids


def test_best_per_record_and_decide():
    rec = np.array([7, 7, 3, 3, 3, 9])
    prob = np.array([0.2, 0.9, 0.5, 0.8, 0.8, 0.3], np.float32)
    assert best_per_record(rec, prob).tolist() == [False, True, False, True, False, True]
    assert decide(rec, prob, 0.5).tolist() == [False, True, False, True, False, False]


def test_vectorized_f05_matches_reference():
    rng = np.random.default_rng(0)
    b_ids = [f"S1-{i}" for i in range(1, 301)]
    truth = {s: {f"S2-{i * 10 + j}" for j in range(rng.integers(0, 4))} for i, s in enumerate(b_ids, 1)}
    rows = []
    for i, s in enumerate(b_ids, 1):  # true pairs plus look-alike negatives, some pointing at non-B S1s
        rows += [(s, r, 1) for r in truth[s]]
        rows += [(s, f"S3-{i * 10 + j}", 0) for j in range(rng.integers(0, 3))]
        rows += [("S1-999999", f"S2-{i * 10}", 0)]
    pred = pd.DataFrame(rows, columns=["s1_id", "cand_id", "label"])
    pred["prob"] = rng.random(len(pred)).astype(np.float32)
    pred["query_group"] = np.where(rng.random(len(pred)) < 0.3, "other_sampled", "target")
    ev = Eval(pred, b_ids, truth, pd.Series("us", index=b_ids))
    for t in (0.0, 0.3, 0.7, 1.01):  # expected matches built independently: best S1 per record, prob >= t
        best = pred.loc[pred.groupby("cand_id")["prob"].idxmax()]
        best = best[(best["prob"] >= t) & best["s1_id"].isin(b_ids)]
        want = best.groupby("s1_id")["cand_id"].apply(set).to_dict()
        assert abs(ev.score(t)["f05"] - macro_f05(want, truth, b_ids)) < 1e-12
        assert ev.matches(t) == want
    assert ev.score(0.3)["f05_corrected"] <= ev.score(0.3)["f05"]


def test_f05_counts_special_cases():
    got = f05_per_s1(np.array([0, 0, 0, 2]), np.array([0, 0, 3, 0]), np.array([0, 1, 0, 0]))
    assert got.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_write_grouped(tmp_path):
    s1_all = id_key(["S1-1", "S1-2", "S1-3"])
    write_grouped(tmp_path / "m.tsv", ["source1_entity_id", "matched_entity_ids"], s1_all,
                  id_key(["S1-3", "S1-1", "S1-3"]), id_key(["S2-9", "S3-4", "S2-1"]))
    assert (tmp_path / "m.tsv").read_text() == ("source1_entity_id\tmatched_entity_ids\n"
                                                "S1-1\tS3-4\nS1-2\t\nS1-3\tS2-1,S2-9\n")
