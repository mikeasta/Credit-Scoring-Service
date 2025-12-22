import mlflow
import logging
import pandas as pd
from typing import Literal
from data import load_train_data, get_train_data, \
    split_train_data, smote_data, get_catboost_pools


def run_study(
    client_type: str | Literal["new_client", "old_client"] = "new_client"
) -> None:
    """Runs whole experiment tracking process"""
    mlflow.set_experiment("MLFlow study")
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run():
        data = load_train_data()
        features, targets = get_train_data(
            data=data, 
            client_type=client_type,
            ohe=False
        )

        X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_data(
            features=features,
            targets=targets,
        )

        X_train, y_train = smote_data(
            features=X_train,
            targets=y_train,
            client_type=client_type
        )

        



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_study()