import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

class ClientData(BaseModel):
    age: int
    education: int
    income: int
    has_car: int | bool
    has_passport: int | bool
    has_work: int | bool

app = FastAPI()
classifier = joblib.load("./models/best_recall_model.joblib")

@app.post("/score")
def score(data: ClientData):
    features = [
        data.age,
        data.education,
        data.income,
        data.has_car,
        data.has_passport,
        data.has_work
    ]
    X = np.array([features])
    default = classifier.predict(X).item()  
    approved = not default
    return { "approved": approved }
    