import pytest

from src.metric import f05_entity, macro_f05


def test_problem_statement_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert f05_entity(pred, true) == pytest.approx(0.714, abs=5e-4)


def test_singleton_cases():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"S2-1"}, set()) == 0.0
    assert f05_entity(set(), {"S2-1"}) == 0.0


def test_guide_table():
    assert f05_entity({"A"}, {"A", "B", "C"}) == pytest.approx(0.714, abs=5e-4)
    assert f05_entity({"A", "B", "C", "X"}, {"A", "B", "C"}) == pytest.approx(0.789, abs=5e-4)
    assert f05_entity({"X"}, {"A"}) == 0.0


def test_macro_average_and_missing_pred():
    truth = {"a": set(), "b": {"S2-1"}, "c": {"S2-2"}}
    pred = {"b": {"S2-1"}}  # a: empty -> 1.0, b: exact -> 1.0, c: missing -> 0.0
    assert macro_f05(pred, truth, ["a", "b", "c"]) == pytest.approx(2 / 3)
