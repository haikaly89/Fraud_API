from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel

with open("model/train_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

class InputData(BaseModel):
    amount: float
    merchant: str
    device_type: str

app = FastAPI(title="Fraud Detection API - Low Risk Tuning")

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API (Low Risk Tuning) is running"}

@app.post("/predict")
def predict(data: InputData):

    sample_df = pd.DataFrame([data.dict()])

    sample_df["amount_log"] = np.log1p(sample_df["amount"])

    combined = pd.DataFrame(columns=["merchant", "device_type"])
    combined = pd.concat([combined, sample_df[["merchant", "device_type"]]], ignore_index=True)

    combined_encoded = pd.get_dummies(combined, drop_first=True)

    sample_encoded = combined_encoded.iloc[[-1]].reset_index(drop=True)

    numerical_cols = ["amount_log"]
    sample_final = pd.concat([sample_df[numerical_cols].reset_index(drop=True), sample_encoded], axis=1)

    for col in model_columns:
        if col not in sample_final.columns:
            sample_final[col] = 0

    sample_final = sample_final[model_columns]

    prediction = model.predict(sample_final)
    proba = model.predict_proba(sample_final)

    return {
        "prediction": int(prediction[0]),
        "label": "Fraud" if prediction[0] == 1 else "Non-Fraud",
        "probability": {
            "Non-Fraud": float(proba[0][0]),
            "Fraud": float(proba[0][1])
        }
    }
