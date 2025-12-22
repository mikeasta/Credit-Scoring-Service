import yaml
import pathlib
import logging
import pandas as pd
from catboost import Pool
from imblearn.over_sampling import SMOTENC
from typing import List, Tuple, Dict, Literal
from sklearn.model_selection import train_test_split

def _load_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Loads CSV data in DataFrame format from specific path"""
    data = pd.read_csv(path)
    return data


def load_train_data() -> pd.DataFrame:
    """Loads processed train dataser"""
    return _load_data("../data/processed/train_dataset.csv")


def load_yaml_config(path: str | pathlib.Path) -> Dict:
    """Loads YAML config as Python dictionary"""
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
        return config


def convert_dataframe_into_series(df: pd.DataFrame) -> pd.Series:
    """Converts single-columns dataframe into Series"""
    assert len(df.columns) == 1, f"Too much columns in dataframe: {df.columns}"
    return df[df.columns[0]]


def get_train_data(
    data: pd.DataFrame, 
    client_type: str | Literal["new_client", "old_client"],
    ohe: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits train data into features and targets
    
    Returns: 
    - features: features dataframe
    - targets: targets dataframe
    """
    # Load data
    data_config = load_yaml_config("../configs/data.yaml")
    client_config = data_config.get(client_type, None)
    assert client_config, f"There is no '{client_type}' client type"

    # Pick columns
    assert len(client_config["targets"]) == 1,\
        f"Too much target columns: {client_config["targets"]}"

    targets = data[client_config["targets"]]
    features = data[client_config["features"]]

    # One-Hot-Encoding
    if ohe:
        features = pd.get_dummies(features, columns=client_config["cat"])

    return (features, targets)


def split_train_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    stratify: bool = True,
    random_state: int = 42
) -> List[pd.DataFrame]:
    """
    Splits train data into train, test and valid datasets

    Returns:
    X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    data_config = load_yaml_config("../configs/data.yaml")["train_data_split"]
    test_size = data_config["test_size"]
    validation_size = data_config["validation_size"]
    assert test_size > 0 and validation_size > 0,\
        f"You need to pick split size larger than 0.\n \
        Current split sizes: {test_size=}, {validation_size=}"

    # Check if we chose to stratify
    stratify = targets.copy() if stratify else None

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    stratify = y_train_valid.copy() if stratify is not None else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify
    )

    return (
        X_train, 
        X_valid, 
        X_test, 
        convert_dataframe_into_series(y_train),
        convert_dataframe_into_series(y_valid),
        convert_dataframe_into_series(y_test)
    )


def log_target_distribution(
    targets: pd.Series | pd.DataFrame
) -> None:
    """Logs target distribution"""
    logging.info("Target distribution:")
    unique_values = targets.unique()
    for value in unique_values: 
        logging.info(f"Class {value}: {sum(targets == value)}")
        logging.info(f"Ratio of {value}: {sum(targets == value) / len(targets) * 100:.1f}%\n")
    else:
        logging.info(f"Total: {len(unique_values)} classes, {len(targets)} samples")


def smote_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    client_type: str | Literal["new_client", "old_client"],
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs SMOTE over input data"""
    # Getting data configs
    data_config = load_yaml_config("../configs/data.yaml")
    client_config = data_config.get(client_type, None)
    assert client_config, f"There is no '{client_type}' client type"

    # Log pre-SMOTE target distribution
    log_target_distribution(targets)

    # Transforms categorical variables into features column indices
    cat_indices = [features.columns.to_list().index(f) for f in client_config["cat"]]

    # Transform data
    smote_nc = SMOTENC(
        categorical_features=cat_indices, 
        random_state=random_state
    )

    features, targets = smote_nc.fit_resample(features, targets)

    # Log post-SMOTE target distribution
    log_target_distribution(targets)
    
    return (features, targets)


def get_catboost_pools(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    client_type: str | Literal["new_client", "old_client"]
) -> Tuple[Pool]:
    """
    Returns CatBoost data pools

    Returns:
    train_pool, valid_pool, test_pool
    """
    data_config = load_yaml_config("../configs/data.yaml")
    client_config = data_config.get(client_type, None)
    assert client_config, f"There is no '{client_type}' client type"

    cat_features = client_config["cat"]
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)

    return (train_pool, valid_pool, test_pool)


def get_model_params(model_name: str) -> Dict:
    models_config = load_yaml_config("../configs/models.yaml")
    model_config = models_config.get(model_name, None)
    assert model_config, f"There is no '{model_name}' model"

    return model_config