import joblib
import catboost
from pathlib import Path
from models import build_model
from data import load_yaml_config, get_train_data
from sklearn.model_selection import train_test_split

_MODEL_PARAMETERS = {
    "new_client": {
        "source": Path("../models/best_catboost_new_client_model.joblib"),
        "destination": Path("../models/best_catboost_new_client_model_v1.cbm")
    },
    "old_client": {
        "source": Path("../models/best_catboost_old_client_model.joblib"),
        "destination": Path("../models/best_catboost_old_client_model_v1.cbm")
    }
}

# Configuration paths
_DATA_CONFIGS = Path("../configs/data.yaml")


def main():
    """Train CatBoost models using optimized hyperparameters"""
    # Import configs
    data_configs = load_yaml_config(_DATA_CONFIGS)

    for client_type, paths in _MODEL_PARAMETERS.items():
        # Load data and parameters
        features, targets = get_train_data(client_type=client_type)
        cat_features = data_configs[client_type]["cat"] 
        params = joblib.load(paths["source"])

        # Build model
        model = build_model("catboost", params)

        # Train model
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            targets,
            test_size=0.1,
            stratify=targets
        )

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
        model.save_model(paths["destination"])

if __name__ == "__main__":
    main()