import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve
)


def evaluate_all(
    y_true: pd.DataFrame | pd.Series | List[int],
    y_proba: pd.DataFrame | pd.Series | List[float],
    threshold: float 
) -> Dict[str, float]:
    """Evaluates all metrics"""
    y_pred = (y_proba > threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def expected_cost(
    y_true: pd.Series | List | pd.DataFrame, 
    y_proba: pd.Series | List | pd.DataFrame, 
    threshold: float,
    C_fp: float,
    C_fn: float,
) -> float:
    """Returns expected cost of money losts"""
    y_pred = (y_proba >= threshold).astype(int)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return C_fp * fp + C_fn * fn


def optimize_threshold(
    y_true: pd.Series | List | pd.DataFrame,
    y_proba: pd.Series | List | pd.DataFrame,
    C_fp: float,
    C_fn: float,
) -> float:
    """
    Finds best threshold according to input data by optimization precision-recall curve.
    """
    # Get PRC keypoints
    prec, rec, ths = precision_recall_curve(y_true, y_proba)

    # Adds 1.0 threshold
    ths = np.append(ths, 1.0)

    # Looking for best threshold
    best_threshold, best_cost = 0.5, expected_cost(y_true, y_proba, 0.5, C_fp, C_fn)
    for t in ths:
        cost = expected_cost(y_true, y_proba, t, C_fp, C_fn)
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold