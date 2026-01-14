import yaml
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from database import get_record_by_name
from utils import (
    get_classifier, 
    process_client_data, 
    predict_default,
    create_result_message
)

# FastAPI Setup
class ClientData(BaseModel):
    name_surname: str
    age: int
    education: int
    income: int
    has_car: int | bool
    has_work: int | bool
    has_passport: int | bool

app = FastAPI()


# === ENDPOINTS ===

@app.post("/score")
def score(data: ClientData):
    """
    Scores client according to his info and returns credit-scoring system decision result
    """

    # Check if client in database
    client_data = get_record_by_name(data.name_surname)
   
    # Get prediction model
    # client_data is always a DataFrame (empty if error or not found)
    client_present = client_data is not None and len(client_data) == 1
    classifier = get_classifier(
        client_type="old_client" if client_present else "new_client"
    )

    # Prepare input data to suit model input
    features_input = process_client_data(
        form_data=data, 
        client_data=client_data if client_present else None
    )
    
    # Make prediction
    default_proba = classifier.predict_proba(features_input)[1]
    default = predict_default(default_proba, client_present)

    # Create result message
    message = create_result_message(
        client_present=client_present,
        approved=not default,
        name_surname=data.name_surname
    )

    return { "message": message }


@app.get("/greet")
def greet():
    """
    Test message for fast and convenient troubleshooting
    """
    return { "message": "Hello World!" }
    