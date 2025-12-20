from typing import Literal, Dict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

_MODEL_NAMES = [
    "xgb",
    "lightgbm",
    "tabpfn",
    "catboost",
    "logreg",
    "srf", 
    "sgb"
]

def build_model(name: Literal[*_MODEL_NAMES], params: Dict) -> object:
    """
    Creates classifier model

    Supported models:
    - `xgb` - XGBoost classifier
    - `lightgbm` - LightGBM classifier
    - `tabpfn` - TabPFN classifier (pre-trained neural network)
    - `catboost` - CatBoost classifier
    - `logreg` - Logistic Regression classifier
    - `srf` - Scikit-Learn Random Forest classifier
    - `sgb` - Scikit-Learn Gradient Boosting classifier

    Returns:
    - Initialized with `params` classifier model
    """
    match (name):
        case "xgb": 
            return XGBClassifier(**params)
        case "lightgbm": 
            return LGBMClassifier(**params)
        case "tabpfn": 
            return TabPFNClassifier(**params)
        case "catboost": 
            return CatBoostClassifier(**params)
        case "logreg": 
            return LogisticRegression(**params)
        case "srf": 
            return RandomForestClassifier(**params)
        case "sgb": 
            return GradientBoostingClassifier(**params)
        case _: 
            raise ValueError(f"There is no model '{name}'.")