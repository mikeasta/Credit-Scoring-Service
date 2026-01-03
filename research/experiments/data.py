import yaml
import pathlib
import pandas as pd
from typing import Tuple, Dict, Literal
from sklearn.model_selection import StratifiedKFold


def _load_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Loads CSV data in DataFrame format from specific path"""
    data = pd.read_csv(path, index_col="id")
    return data


def load_yaml_config(path: str | pathlib.Path) -> Dict:
    """Loads YAML config as Python dictionary"""
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
        return config


def get_train_data(
    client_type: str | Literal["new_client", "old_client"],
    ohe: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits train data into features and targets
    
    Returns: 
    - features: features dataframe
    - targets: targets series
    """
    # Load data
    data = _load_data("../data/processed/train_dataset.csv")

    # Apply client config settings to data
    data_config = load_yaml_config("../configs/data.yaml")
    client_config = data_config.get(client_type, None)
    assert client_config, f"There is no '{client_type}' client type"

    # Pick consequent columns
    targets = data[client_config["target"]]
    features = data[client_config["features"]]

    # One-Hot-Encoding (if necessary)
    if ohe:
        features = pd.get_dummies(features, columns=client_config["cat"])

    return (features, targets)


def get_cv(
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: int = 42
) -> StratifiedKFold:
    """
    Generates StratifiedKFold instance that splits data into cross-validation folds

    Create splits using:
    ```python
    skf=StratifiedKFol(n_splits=5)
    skf.get_n_splits()
    >>> 5
    ```

    Then you can iterate over folds using for-loop:
    ```python
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        ...
    ```
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)