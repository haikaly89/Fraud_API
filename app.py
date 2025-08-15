from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd #type: ignore
import numpy as np
import pickle
import tensorflow as tf #type: ignore
import uvicorn
import io
from typing import List

# --- Memuat Model dan Aset ---
# Pastikan file model dan scaler berada di direktori yang benar
try:
    model = tf.keras.models.load_model("model/fraud_model_tf.h5")
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    print("Model dan aset berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat aset: {e}")
    # Sebaiknya hentikan aplikasi jika model gagal dimuat
    exit()

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI(title="Fraud Detection API",
              description="API untuk mendeteksi fraud transaksi tunggal dan batch dari file CSV.")

# --- Pydantic Model untuk Input JSON Tunggal ---
class TransactionData(BaseModel):
    amount: float
    merchant_type: str
    device_type: str

# --- Endpoint untuk Prediksi Transaksi Tunggal (JSON) ---
@app.post("/predict", tags=["Single Prediction"])
def predict(data: TransactionData):
    """
    Menerima satu data transaksi dalam format JSON dan mengembalikan prediksi fraud.
    """
    try:
        # Mengubah input Pydantic menjadi DataFrame
        input_df = pd.DataFrame([data.dict()])

        # --- Preprocessing ---
        # 1. Transformasi log pada 'amount'
        input_df['amount_log'] = np.log1p(input_df['amount'])
        
        # 2. One-hot encoding untuk fitur kategorikal
        input_encoded = pd.get_dummies(input_df, columns=['merchant_type', 'device_type'])
        
        # 3. Hapus kolom 'amount' asli
        input_encoded = input_encoded.drop(columns=['amount'], errors='ignore')

        # 4. Menyelaraskan kolom dengan yang digunakan saat training
        final_df = input_encoded.reindex(columns=model_columns, fill_value=0)

        # 5. Scaling fitur
        input_scaled = scaler.transform(final_df)

        # --- Prediksi ---
        prob_fraud = model.predict(input_scaled)[0][0]
        prob_nonfraud = 1 - prob_fraud
        label = "Fraud" if prob_fraud >= 0.5 else "Non-Fraud"

        # --- Mengembalikan Hasil ---
        return {
            "prediction": int(prob_fraud >= 0.5),
            "label": label,
            "probability": {
                "Non-Fraud": float(prob_nonfraud),
                "Fraud": float(prob_fraud)
            }
        }
    except Exception as e:
        return {"error": f"Terjadi kesalahan saat prediksi: {str(e)}"}

# --- Endpoint untuk Prediksi Batch dari File CSV ---
@app.post("/predict_csv", tags=["Batch Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Menerima file CSV, melakukan prediksi untuk setiap baris, dan mengembalikan hasilnya.
    Pastikan kolom di CSV sama dengan input: 'amount', 'merchant_type', 'device_type'.
    """
    try:
        # Membaca konten file yang diunggah
        contents = await file.read()
        
        # Mengubah konten menjadi DataFrame pandas
        # Menggunakan io.StringIO untuk membaca string sebagai file
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if not all(col in df.columns for col in ['amount', 'merchant_type', 'device_type']):
            return {"error": "File CSV harus memiliki kolom: 'amount', 'merchant_type', 'device_type'"}

        # --- Preprocessing untuk seluruh DataFrame ---
        df_processed = df.copy()
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_encoded = pd.get_dummies(df_processed, columns=['merchant_type', 'device_type'])
        df_encoded = df_encoded.drop(columns=['amount'], errors='ignore')
        final_df = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Scaling fitur untuk seluruh data
        input_scaled = scaler.transform(final_df)

        # --- Prediksi untuk seluruh data ---
        predictions = model.predict(input_scaled)

        # --- Format Hasil ---
        results = []
        for index, row in df.iterrows():
            prob_fraud = predictions[index][0]
            prob_nonfraud = 1 - prob_fraud
            label = "Fraud" if prob_fraud >= 0.5 else "Non-Fraud"
            
            result = {
                "original_data": row.to_dict(),
                "prediction": int(prob_fraud >= 0.5),
                "label": label,
                "probability": {
                    "Non-Fraud": float(prob_nonfraud),
                    "Fraud": float(prob_fraud)
                }
            }
            results.append(result)

        return results

    except Exception as e:
        return {"error": f"Gagal memproses file: {str(e)}"}

# --- Menjalankan Server (jika file ini dieksekusi langsung) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
