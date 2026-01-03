import optuna
import mlflow
import numpy as np
import pandas as pd
from catboost import Pool
from typing import Dict, List, Callable

from data import get_cv
from models import build_model, get_proba
from metrics import optimize_threshold, evaluate_all


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
                    spec["choices"]
                )
            case _: 
                raise ValueError(f"There is no variable type'{t}'")

    return params


def objective_factory(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    cat_features: List | pd.Series | pd.Index,
    model_name: str,
    search_space: Dict,
    cv_config: Dict,
    costs_config: Dict,
    target_metric: str,
    mlflow_target_run_id: str = None
) -> Callable:
    """Returns objective function for Optuna experiment running"""
    cv = get_cv(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"]
    )

    def objective(trial: optuna.Trial):
        params = suggest_params(trial=trial, search_space=search_space)

        with mlflow.start_run(
            run_name=f"{model_name}.trial_{trial.number}",
            nested=True
        ):
            mlflow.log_params({f"{model_name}.{k}": v for k, v in params.items()})
            
            # All classification metrics over cross-validation
            metrics_folds = []

            # Stratified cross-validation iterable
            cv_iterable = cv.split(features, targets)
            for train_idx, valid_idx in cv_iterable:
                # Data split
                X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
                y_train, y_valid = targets.iloc[train_idx], targets.iloc[valid_idx]

                # Model building and fitting
                model = build_model(model_name, params)
                if model_name == "catboost":
                    # CatBoost requires special data preparation and fitting steps
                    train_pool = Pool(X_train, y_train, cat_features=cat_features)
                    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
                    model.fit(
                        train_pool, 
                        eval_set=valid_pool, 
                        early_stopping_rounds=100,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)

                # Prediction logits
                y_proba = get_proba(
                    model=model, 
                    features=X_valid, 
                    model_name=model_name,
                    cat_features=cat_features 
                )

                # Looking for optimal threshold
                optimal_threshold = optimize_threshold(
                    y_true=y_valid,
                    y_proba=y_proba,
                    C_fp=costs_config["C_fp"],
                    C_fn=costs_config["C_fn"]
                )

                # Calculate metrics using optimal threshold
                metrics_record = evaluate_all(
                    y_true=y_valid,
                    y_proba=y_proba,
                    threshold=optimal_threshold
                )

                metrics_folds.append(metrics_record)


            # Aggregate metrics over whole cross-validation
            # Calculating mean values for each metric
            agg_metrics = {
                k: float(
                    np.mean([
                        mf[k] for mf in metrics_folds
                    ])
                ) for k in metrics_folds[0].keys()
            }

            # Log metrics
            mlflow.log_metrics({f"{model_name}.{k}": v for k, v in agg_metrics.items()})

            # Log metrics into target (parent) runs 
            if mlflow_target_run_id:
                client = mlflow.tracking.MlflowClient()
                for k, v in agg_metrics.items():
                    client.log_metric(
                        run_id=mlflow_target_run_id,
                        key=k,
                        value=v,
                        step=trial.number+1
                    )

            # Saving aggregated metrics in trial instance
            trial.set_user_attr("agg_metrics", agg_metrics)

            # Optimizing (maximizing) target metric
            return agg_metrics.get(target_metric, agg_metrics["pr_auc"])

    return objective