import pathlib
import pandas as pd
from typing import Tuple, Iterable
from sklearn.model_selection import StratifiedKFold

def load_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Loads CSV data in DataFrame format from specific path"""
    data = pd.read_csv(path)
    return data


def get_train_data(
    data: pd.DataFrame, 
    target_col: str | Iterable 
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits train data into features and targets
    
    Returns: 
    - features: features dataframe
    - targets: targets dataframe
    """
    target_cols = [target_col] if isinstance(target_col, str) else target_col

    for target in target_cols:
        if target not in data.columns:
            raise Exception(f"There is no '{target_col}' feature")

    features = data.drop(columns=[target_col], axis=1)
    targets = data[target_col]
    return (features, targets)
