import optuna
import pandas as pd
from typing import Dict, Literal, List, Callable
from .models import build_model

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
    cat_features: List | pd.Series | pd.Index,
    optimize_metric: str
) -> Callable:
    """Returns objective function for Optuna experiment running"""

    def objective(trial: optuna.Trial):
        params = suggest_params(trial=trial, search_space=search_space)
        model = build_model(params)






    return objective