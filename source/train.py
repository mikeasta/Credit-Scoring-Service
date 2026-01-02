import joblib
import catboost
from pathlib import Path
from models import build_model
from data import load_yaml_config, get_train_data
from sklearn.model_selection import train_test_split

_MODEL_PARAMETERS = {
<<<<<<< HEAD
    "new_client": Path("../artifacts/models/best_catboost_new_client_model.joblib"),
    "old_client": Path("../artifacts/models/best_catboost_old_client_model.joblib"),
}

_SAVE_PATHS = {
    "new_client": Path("../artifacts/models/best_catboost_new_client_model_v1.cbm"),
    "old_client": Path("../artifacts/models/best_catboost_old_client_model_v1.cbm"),
=======
    # "new_client": Path("../artifacts/models/best_catboost_new_client_model.joblib"),
    # "old_client": Path("../artifacts/models/best_catboost_old_client_model.joblib"),
    "new_client_srf": Path("../artifacts/models/best_srf_new_client_recall_model.joblib")
}

_SAVE_PATHS = {
    # "new_client": Path("../artifacts/models/best_catboost_new_client_model_v1.cbm"),
    # "old_client": Path("../artifacts/models/best_catboost_old_client_model_v1.cbm"),
    "new_client_srf": Path("../artifacts/models/best_srf_new_client_recall_model_weights.joblib")
>>>>>>> 5b137d4fb37a106d9c8a7eae6c4bc602763d2693
}

# Configuration paths
_DATA_CONFIGS = Path("../configs/data.yaml")


def main():
    """Train CatBoost models using optimized hyperparameters"""
    # Import configs
    data_configs = load_yaml_config(_DATA_CONFIGS)

    for client_type, params_path in _MODEL_PARAMETERS.items():
        # Load data and parameters
<<<<<<< HEAD
        features, targets = get_train_data(client_type=client_type)
        cat_features = data_configs[client_type]["cat"] 
        params = joblib.load(params_path)

        # Build model
        model = build_model("catboost", params)
=======
        features, targets = get_train_data(client_type="new_client")
        cat_features = data_configs["new_client"]["cat"] 
        params = joblib.load(params_path)

        # Build model
        model = build_model("srf", params)
>>>>>>> 5b137d4fb37a106d9c8a7eae6c4bc602763d2693

        # Train model
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            targets,
            test_size=0.1,
            stratify=targets
        )

<<<<<<< HEAD
        # Catboost
        train_pool = catboost.Pool(X_train, y_train, cat_features)
        valid_pool = catboost.Pool(X_valid, y_valid, cat_features)

        model.fit(
            train_pool, 
            eval_set=valid_pool,
            early_stopping_rounds=100,
            verbose=True
        )

        # Save model
        model.save_model(_SAVE_PATHS[client_type])
=======
        if "srf" in client_type:
            # RandomForest
            model.fit(X_train, y_train)
            joblib.dump(model, _SAVE_PATHS[client_type])

        else:
            # Catboost
            train_pool = catboost.Pool(X_train, y_train, cat_features)
            valid_pool = catboost.Pool(X_valid, y_valid, cat_features)

            model.fit(
                train_pool, 
                eval_set=valid_pool,
                early_stopping_rounds=100,
                verbose=True
            )

            # Save model
            model.save_model(_SAVE_PATHS[client_type])
>>>>>>> 5b137d4fb37a106d9c8a7eae6c4bc602763d2693

if __name__ == "__main__":
    main()