import pandas as pd
from typing import Literal, Dict, Iterable, List
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


_MODEL_CONSTRUCTORS = {
    "xgb": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "tabpfn": TabPFNClassifier,
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
    - `tabpfn` - TabPFN classifier (pre-trained neural network)
    - `catboost` - CatBoost classifier
    - `logreg` - Logistic Regression classifier
    - `srf` - Scikit-Learn Random Forest classifier
    - `sgb` - Scikit-Learn Gradient Boosting classifier

    Returns:
    - Initialized with `params` classifier model
    """
    model_class = _MODEL_CONSTRUCTORS.get(key=name, default=None)
    assert model_class, f"There is no '{name}' model"
    return model_class(**params)


def get_proba(
    model,
    X: pd.DataFrame,
    model_name: str,
    cat_features: Iterable[str] = None
) -> pd.Series | List[float] :
    """
    Returns model estimative probabilities of predicted targets
    """
    assert model_name in _MODEL_CONSTRUCTORS, f"There is no '{model_name}' model"

    if model_name == "catboost":
        pool = Pool(data=X, cat_features=cat_features) if cat_features else Pool(X)
        return model.predict_proba(pool)[:, 1]
    else:
        return model.predict_proba(X)[:, 1]