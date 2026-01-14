# Utilities for FastAPI server
import yaml
import pandas as pd
from typing import Literal
from pathlib import Path
from catboost import CatBoostClassifier


def load_yaml_config(path: str | Path) -> dict:
    """Loads YAML config as Python dictionary"""
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
        return config


def get_classifier(client_type: str | Literal["new_client", "old_client"]) -> CatBoostClassifier:
    """
    Returns classifier (CatBoost) object
    
    :param client_type: For new or familiar (to bank) clients there are different models. 
    :type client_type: str | Literal["new_client", "old_client"]
    :return: Classifier object
    :rtype: CatBoostClassifier
    """
    # Get path relative to this file's directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "server.yaml"
    paths_config = load_yaml_config(config_path)["paths"]

    # Save model path
    model_path = str(base_dir / "models" / paths_config[client_type])

    # Create model and load weights
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def process_client_data(
    form_data: dict | pd.DataFrame, 
    client_data: dict | pd.DataFrame = None
):
    """Processes client data and returns features input"""
    features = []
    if client_data is None:
        features = [
            form_data.age,
            form_data.education,
            form_data.income,
            form_data.has_car,
            form_data.has_work,
            form_data.has_passport
        ]
    else:
        client_data = client_data.iloc[0].to_dict()
        features = [
            form_data.age,
            form_data.education,
            form_data.income,
            form_data.has_car,
            client_data["car_is_foreign"] if form_data.has_car else 0,
            form_data.has_work,
            form_data.has_passport,
            client_data["bki_score"],
            client_data["requests_count"],
            client_data["rejected_applications_count"],
            client_data["region_rating"],
            client_data["home_address_category"],
            client_data["work_address_category"],
            client_data["social_network_analysis_score"],
            client_data["first_record_age"],
        ]

    return features


def predict_default(
    default_probability_estimation: float,
    client_present: bool
):
    """
    Predicts client default using suitable classification threshold according to client type
    """
    # Load model thresholds
    base_dir = Path(__file__).parent
    config_path = base_dir / "server.yaml"
    classifier_config = load_yaml_config(config_path)["classifier"]

    # Make decision
    threshold_type = "old_client_threshold" if client_present else "new_client_threshold"
    threshold = classifier_config[threshold_type]
    return (default_probability_estimation > threshold)


def create_result_message(
    client_present: bool,
    approved: bool,
    name_surname: str = None
) -> str:
    """Creates scoring desicion message"""
    # Choose greet variant according to client type.
    greet = ""
    if client_present and name_surname:
        greet = f"С возвращением, {name_surname}."
    elif not client_present and name_surname:
        greet = f"Добро пожаловать, {name_surname}!"
    else:
        greet = "Добро пожаловать"

    # Choose decision variant according to default classification result
    decision = ""
    if approved:
        decision = "Ваша заявка одобрена"
    else:
        decision = "Мы подобрали для вас альтернативный вариант - карта с 5% кэшбеком во всех супермаркетах Санкт-Петербурга"

    return " ".join([greet, decision])
