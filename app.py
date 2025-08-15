from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd #type: ignore
import numpy as np # Import numpy
import pickle
import tensorflow as tf #type: ignore
import uvicorn

try:
    model = tf.keras.models.load_model("model/fraud_model_tf.h5")
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    print("Model dan aset berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat aset: {e}")
    exit()

app = FastAPI(title="Fraud Detection API")

class TransactionData(BaseModel):
    amount: float
    merchant_type: str  
    device_type: str

@app.post("/predict")
def predict(data: TransactionData):
    input_df = pd.DataFrame([data.dict()])

    input_df['amount_log'] = np.log1p(input_df['amount'])

    input_encoded = pd.get_dummies(input_df, columns=['merchant_type', 'device_type'])
    
    input_encoded = input_encoded.drop(columns=['amount'], errors='ignore')

    final_df = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(final_df)

    prob_fraud = model.predict(input_scaled)[0][0]
    prob_nonfraud = 1 - prob_fraud
    label = "Fraud" if prob_fraud >= 0.5 else "Non-Fraud"

    return {
        "prediction": int(prob_fraud >= 0.5),
        "label": label,
        "probability": {
            "Non-Fraud": float(prob_nonfraud),
            "Fraud": float(prob_fraud)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)