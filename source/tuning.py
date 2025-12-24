from lightgbm import early_stopping
import optuna
import mlflow
import pandas as pd
from catboost import Pool
from typing import Dict, Literal, List, Callable

from .data import get_cv
from .models import build_model, get_proba


def suggest_params(
    trial: optuna.Trial, 
    search_space: Dict
) -> Dict:
    """Returns Optuna.trial parameters for model optimization"""
    params = {}

    for key, spec in search_space.items():
        t = spec["type"]

        match (t):
            case "int": 
                params[key] = trial.suggest_int(
                    key, 
                    spec["low"], 
                    spec["high"]
                )
            case "float": 
                params[key] = trial.suggest_float(
                    key, 
                    spec["low"], 
                    spec["high"], 
                    log=spec.get("log", False)
                )
            case "cat": 
                params[key] = trial.suggest_categorical(
                    key, 
                    spec["choises"]
                )
            case _: 
                raise ValueError(f"There is no variable type'{t}'")

    return params


def objective_factory(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    model_name: str | Literal["new_client", "old_client"],
    search_space: Dict,
    cv_config: Dict,
    cat_features: List | pd.Series | pd.Index,
    optimize_metric: str
) -> Callable:
    """Returns objective function for Optuna experiment running"""
    cv = get_cv(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"]
    )

    def objective(trial: optuna.Trial):
        params = suggest_params(trial=trial, search_space=search_space)

        with mlflow.start_run(nested=True):
            mlflow.log_params({f"{model_name}.{k}": v for k, v in params.items()})
            
            # All classification metrics over cross-validation
            metrics_folds = []

            # Stratified cross-validation iterable
            cv_iterable = cv.split(features, targets)
            for fold_idx, (train_idx, valid_idx) in enumerate(cv_iterable):
                # Data split
                X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
                y_train, y_valid = targets.iloc[train_idx], features.iloc[valid_idx]

                # Model building and fitting
                model = build_model(params)
                if model_name == "catboost":
                    # CatBoost requires special data preparation and fitting steps
                    train_pool = Pool(X_train, y_train, cat_features=cat_features)
                    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
                    model.fit(
                        train_pool, 
                        eval_set=valid_pool, 
                        early_stopping_rounds=search_space["early_stopping_rounds"],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)

                y_proba = get_proba(
                    model=model, 
                    features=X_valid, 
                    model_name=model_name,
                    cat_features=cat_features 
                )




    return objective