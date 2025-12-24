import optuna
import mlflow
import logging
from typing import Literal

from data import (
    load_yaml_config,
    get_train_data
)
from tuning import objective_factory


def run_study(
    client_type: str | Literal["new_client", "old_client"] = "new_client"
) -> None:
    """Runs whole experiment tracking process"""
    # Import configs
    data_configs = load_yaml_config("../configs/data.yaml")
    models_configs = load_yaml_config("../configs/models.yaml")
    experiment_configs = load_yaml_config("../configs/experiment.yaml")

    # Load data
    features, targets = get_train_data(client_type=client_type)
    cat_features = data_configs[client_type]["cat"]

    # Setting up MLFlow 
    mlflow.set_tracking_uri(experiment_configs["logging"]["mlflow_uri"])

    # Create experiment if it doesn't exist, or get existing one
    experiment_name = experiment_configs["logging"]["mlflow_experiment"]
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        pass  # Experiment might already exist
    
    mlflow.set_experiment(experiment_name)
    
    # Running experiments
    results = []
    for model_name, model_spec in models_configs.items():
        logging.info(f"Running Optuna for: {model_name}")
        with mlflow.start_run(run_name=f"{model_name}_optuna"):
            objective = objective_factory(
                features=features,
                targets=targets,
                cat_features=cat_features,
                model_name=model_name,
                search_space=model_spec["search_space"],
                cv_config=experiment_configs["cv"],
                costs_config=experiment_configs["costs"],
                target_metric=experiment_configs["metrics"]["optimize"]
            )

            # Optimization process
            study = optuna.create_study(direction="maximize")
            study.optimize(
                func=objective, 
                n_trials=model_spec.get("n_trials", 40)
            )

            # Getting best metrics
            best_metrics = study.best_trial.user_attrs["agg_metrics"]

            # Logging best hyperparams and metrics
            mlflow.log_params({f"{model_name}.best_params": study.best_params})
            mlflow.log_metrics({f"{model_name}.best_{k}": v for k, v in best_metrics.items})

            # Saving overall info
            results.append((model_name, best_metrics, study.best_params))

    # Picking best model
    key = experiment_configs["metrics"]["optimize"]
    best = max(results, key=lambda x: x[1][key])
    best_model_name, best_metrics, _ = best
    logging.info(f"Best model: {best_model_name}; {key}: {best_metrics[key]}")
    return best


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    mlflow.enable_system_metrics_logging()
    run_study()
