import mlflow
import logging
from typing import Literal


# TODO: Запуск экспериментов (Optuna.study)
def run_study(
    client_type: str | Literal["new_client", "old_client"] = "new_client"
) -> None:
    """Runs whole experiment tracking process"""
    mlflow.set_experiment("MLFlow study")
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run():
        pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_study()
