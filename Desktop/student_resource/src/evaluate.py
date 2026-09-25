import pandas as pd
from src.config import BETA

def compute_f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """
    Computes F-beta score. For beta=0.5:
    F0.5 = (1 + 0.25) * (P * R) / (0.25 * P + R)
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

def evaluate_matches(ground_truth_dict: dict, predictions_dict: dict) -> dict:
    """
    Computes Precision, Recall, and Macro F0.5 against ground truth.
    Supports NO_MATCH (empty sets).
    """
    tp, fp, fn = 0, 0, 0
    all_s1_ids = set(ground_truth_dict.keys()) | set(predictions_dict.keys())

    for s1_id in all_s1_ids:
        true_m = ground_truth_dict.get(s1_id, set())
        pred_m = predictions_dict.get(s1_id, set())

        true_m = {m for m in true_m if pd.notna(m) and str(m).strip() not in ('', 'nan', 'None')}
        pred_m = {m for m in pred_m if pd.notna(m) and str(m).strip() not in ('', 'nan', 'None')}

        tp += len(true_m & pred_m)
        fp += len(pred_m - true_m)
        fn += len(true_m - pred_m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = compute_f_beta(precision, recall, beta=BETA)

    return {
        'precision': precision,
        'recall': recall,
        'f0.5': f05,
        'tp': tp, 'fp': fp, 'fn': fn
    }
