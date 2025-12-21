import mlflow
import optuna
import pandas as pd
from typing import Callable
from .data import load_data, get_train_data
from .metrics import evaluate_all
from .models import build_model

def objective_factory(
    model,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    
) -> Callable:
    """
    returns 
    """