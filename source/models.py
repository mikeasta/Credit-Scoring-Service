import pandas as pd
from typing import Literal, Dict, Iterable, List
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from data import load_yaml_config


_MODEL_CONSTRUCTORS = {
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
    "logreg": LogisticRegression,
    "srf": RandomForestClassifier, 
    "sgb": GradientBoostingClassifier
}


def build_model(name: Literal[*_MODEL_CONSTRUCTORS.keys()], params: Dict) -> object:
    """
    Creates classifier model

    Supported models:
    - `xgb` - XGBoost classifier
    - `lightgbm` - LightGBM classifier
    - `catboost` - CatBoost classifier
    - `logreg` - Logistic Regression classifier
    - `srf` - Scikit-Learn Random Forest classifier
    - `sgb` - Scikit-Learn Gradient Boosting classifier

    Returns:
    - Initialized with `params` classifier model
    """
    model_class = _MODEL_CONSTRUCTORS.get(name, None)
    assert model_class, f"There is no '{name}' model"
    return model_class(**params)


def get_proba(
    model,
    features: pd.DataFrame,
    model_name: str,
    cat_features: Iterable[str] = None
) -> pd.Series | List[float] :
    """
    Returns model estimative probabilities of predicted targets
    """
    assert model_name in _MODEL_CONSTRUCTORS, f"There is no '{model_name}' model"

    if model_name == "catboost":
        pool = Pool(data=features, cat_features=cat_features) 
        return model.predict_proba(pool)[:, 1]
    else:
        return model.predict_proba(features)[:, 1]


def get_model_params(model_name: str) -> Dict:
    """Loads and returns model parameters"""
    models_config = load_yaml_config("../configs/models.yaml")
    model_config = models_config.get(model_name, None)
    assert model_config, f"There is no '{model_name}' model"

    return model_config