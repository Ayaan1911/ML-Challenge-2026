"""Exact competition metric (TEAM_GUIDE Section 2.4): F0.5 per S1 entity, macro-averaged."""


def f05_entity(pred_set, true_set) -> float:
    pred, true = set(pred_set or ()), set(true_set or ())
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:  # covers pred empty
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred: dict, truth: dict, s1_ids) -> float:
    """Average over s1_ids; an S1 missing from pred counts as an empty prediction."""
    s1_ids = list(s1_ids)
    if not s1_ids:
        raise ValueError("s1_ids is empty")
    return sum(f05_entity(pred.get(s), truth[s]) for s in s1_ids) / len(s1_ids)
