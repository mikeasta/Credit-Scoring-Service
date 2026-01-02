import yaml
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Any
from database import get_record_by_name
from catboost import CatBoostClassifier

class ClientData(BaseModel):
    name_surname: str
    age: int
    education: int
    income: int
    has_car: int | bool
    has_work: int | bool
    has_passport: int | bool

app = FastAPI()


def get_classifier(client_type: str | Literal["new_client", "old_client"]) -> Any:
    model_path = None
    match(client_type):
        case "new_client": 
            model_path = Path("../artifacts/models/best_catboost_new_client_model_v1.cbm")
        case "old_client": 
            model_path = Path("../artifacts/models/best_catboost_old_client_model_v1.cbm")
        case _: 
            raise Exception(f"There is no such a client type: {client_type}")

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
    

def load_yaml_config(path: str | Path) -> dict:
    """Loads YAML config as Python dictionary"""
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
        return config


def predict_default(
    default_probability_estimation: float,
    client_present: bool
):
    """Predicts client default"""
    # Load model thresholds
    classifier_config = load_yaml_config("../configs/server.yaml")["classifier"]
    threshold_type = "old_client_threshold" if client_present else "new_client_threshold"
    threshold = classifier_config[threshold_type]
    return (default_probability_estimation > threshold)


def create_result_message(
    client_present: bool,
    approved: bool,
    name_surname: str = None
) -> str:
    """Creates scoring desicion message"""
    # Choose greet variant
    greet = ""
    if client_present and name_surname:
        greet = f"С возвращением, {name_surname}."
    elif not client_present and name_surname:
        greet = f"Добро пожаловать, {name_surname}!"
    else:
        greet = "Добро пожаловать"

    # Choose decision variant
    decision = ""
    if approved:
        decision = "Ваша заявка одобрена"
    else:
        decision = "Мы подобрали для вас альтернативный вариант - карта с 5% кэшбеком во всех супермаркетах Санкт-Петербурга"

    return " ".join([greet, decision])


@app.post("/score")
def score(data: ClientData):
    # Check if client in database
    client_data = get_record_by_name(data.name_surname)
   
    # Get prediction model
    client_present = len(client_data) == 1
    classifier = get_classifier(
        client_type="old_client" if client_present else "new_client"
    )

    # Prepare input data
    features_input = process_client_data(
        form_data=data, 
        client_data=client_data if client_present else None
    )
    
    # Make prediction
    default_proba = classifier.predict_proba(features_input)[1]
    default = predict_default(default_proba, client_present)
    approved = not default

    # Create result message
    message = create_result_message(
        client_present=client_present,
        approved=approved,
        name_surname=data.name_surname
    )

    return { "message": message }
    