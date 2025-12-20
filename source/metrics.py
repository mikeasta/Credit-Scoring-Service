import pandas as pd
from typing import List, Dict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def accuracy(
    y_true: pd.DataFrame | pd.Series | List[int],
    y_pred: pd.DataFrame | pd.Series | List[int],
) -> float:
    """Calculates prediction accuracy"""
    return sum(y_true == y_pred) / len(y_pred)


def evaluate_all(
    y_true: pd.DataFrame | pd.Series | List[int],
    y_proba: pd.DataFrame | pd.Series | List[float],
    threshold: float = 0.5
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

