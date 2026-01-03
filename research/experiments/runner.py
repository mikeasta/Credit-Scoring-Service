import optuna
import mlflow
import joblib
import logging
import pandas as pd
from pathlib import Path
from typing import Literal
from tuning import objective_factory
from data import (
    load_yaml_config,
    get_train_data
)


# Configuration paths
_DATA_CONFIGS       = Path("../configs/data.yaml")
_MODELS_CONFIGS     = Path("../configs/models.yaml")
_EXPERIMENT_CONFIGS = Path("../configs/experiment.yaml")


def run_study(
    client_type: str | Literal["new_client", "old_client"] = "new_client"
) -> pd.DataFrame:
    """Runs whole experiment tracking process"""
    # Import configs
    data_configs       = load_yaml_config(_DATA_CONFIGS)
    models_configs     = load_yaml_config(_MODELS_CONFIGS)
    experiment_configs = load_yaml_config(_EXPERIMENT_CONFIGS)

    # Run start
    logging.info(f"Started run for '{client_type}' type. Optimizing {experiment_configs["metrics"]["optimize"]} metric.")
    logging.info(f"Data configuration: {str(_DATA_CONFIGS)}")
    logging.info(f"Models configuration: {str(_MODELS_CONFIGS)}")
    logging.info(f"Experiment configuration: {str(_EXPERIMENT_CONFIGS)}")

    # Load data
    features, targets = get_train_data(client_type=client_type)
    cat_features = data_configs[client_type]["cat"]

    # Setting up MLFlow 
    mlflow_uri = experiment_configs["logging"]["mlflow_uri"]

    # Convert relative path to absolute path if it's a file URI
    if mlflow_uri.startswith("file:"):
        relative_path = mlflow_uri.replace("file:", "").lstrip("/")

        # Get absolute path relative to the source directory
        source_dir = Path(__file__).parent
        absolute_path = (source_dir / relative_path).resolve()

        # Use proper file URI format
        mlflow_uri = absolute_path.as_uri()
    
    mlflow.set_tracking_uri(mlflow_uri)
    logging.info(f"MLflow tracking URI: {mlflow_uri}")

    # Create experiment if it doesn't exist, or get existing one
    experiment_name = experiment_configs["logging"]["mlflow_experiment"]
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logging.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            logging.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        logging.warning(f"Error checking/creating experiment: {e}")
    
    mlflow.set_experiment(experiment_name)
    
    # Running experiments
    results = []
    for model_name, model_spec in models_configs.items():
        logging.info(f"Running Optuna for: {model_name}")
        with mlflow.start_run(run_name=f"{model_name}_optuna") as run:
            objective = objective_factory(
                features=features,
                targets=targets,
                cat_features=cat_features,
                model_name=model_name,
                search_space=model_spec["search_space"],
                cv_config=experiment_configs["cv"],
                costs_config=experiment_configs["costs"],
                target_metric=experiment_configs["metrics"]["optimize"],
                mlflow_target_run_id=run.info.run_id
            )

            # Optimization process
            study = optuna.create_study(direction="maximize")
            study.optimize(
                func=objective, 
                n_trials=model_spec.get("n_trials", 40) if experiment_configs["runner"]["use_n_trials"] else 1
            )

            # Getting best metrics
            best_metrics = study.best_trial.user_attrs["agg_metrics"]

            # Logging best hyperparams and metrics
            mlflow.log_params({f"{model_name}.best_params": study.best_params})
            mlflow.log_metrics({f"{model_name}.best_{k}": v for k, v in best_metrics.items()})

            # Saving overall info
            results.append((model_name, best_metrics, study.best_params))

    # Logging experiment result
    model_names = [x[0] for x in results]
    best_model_metrics = [x[1][experiment_configs["metrics"]["optimize"]] for x in results]

    report_df = pd.DataFrame({
        "Model": model_names,
        f"Best Metric ({experiment_configs["metrics"]["optimize"]})": best_model_metrics
    })

    # Picking best model
    key = experiment_configs["metrics"]["optimize"]
    best = max(results, key=lambda x: x[1][key])
    best_model_name, best_metrics, best_params = best
    logging.info(f"Best model: {best_model_name}; {key}: {best_metrics[key]}")

    # Saving best model params
    joblib.dump(best_params, f"../artifacts/models/best_{best_model_name}_{client_type}_{key}_model.joblib")

    # Return run report
    return report_df


def main(*args, **kwargs) -> None:
    """
    Main function for training experiments
    """
    # Pre-train setup
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    mlflow.enable_system_metrics_logging()

    # Load necessary configurations
    data_configs = load_yaml_config(_DATA_CONFIGS)

    # Running experiments
    reports = []
    for client_type in data_configs.keys():
        report_df = run_study(client_type)
        
        # Saving experiment run report
        reports.append(report_df)

    # Logging 
    for client_type, report_df in zip(data_configs.keys(), reports):
        report_df = report_df.sort_values(by=report_df.columns[1], ascending=False)
        logging.info(f"{client_type} experiment report:\n{report_df.to_string()}")

    # Post-train
    logging.info("Experiment runner finished")    


if __name__ == "__main__":
    main()
