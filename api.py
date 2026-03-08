from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
from mssa_module import mssa_decompose
from typing import List

app = FastAPI()
print("API FILE LOADED")

# =========================
# REQUEST SCHEMA
# =========================
class QoSInput(BaseModel):
    input: List[List[float]]

# =========================
# LOAD MODEL & SCALER
# =========================
model = tf.keras.models.load_model(
    "model_lstm_only.h5",
    compile=False
)

print("MODEL INPUT SHAPE:", model.input_shape)

scalers = joblib.load("scalers.save")

LOOKBACK = 110
MSSA_WINDOW = 50

@app.post("/predict")
def predict(data: QoSInput):

    raw = np.array(data.input, dtype=float)

    print("====================================")
    print("RAW SHAPE:", raw.shape)
    print("RAW LEN:", len(raw))

    # =====================
    # CLEANING
    # =====================
    raw = np.nan_to_num(raw)

    # =====================
    # NORMALISASI
    # =====================
    normalized = np.zeros_like(raw)

    for i in range(4):
        normalized[:, i] = scalers[i].transform(
            raw[:, i].reshape(-1, 1)
        ).flatten()

    # =====================
    # QoS INDEX
    # =====================
    qos = (
        normalized[:, 0] +
        (1 - normalized[:, 1]) +
        (1 - normalized[:, 2]) +
        normalized[:, 3]
    ) / 4

    multivariate = np.column_stack((normalized, qos))

    print("MULTIVARIATE SHAPE:", multivariate.shape)

    # =====================
    # MSSA
    # =====================
    reconstructed = mssa_decompose(multivariate, L=MSSA_WINDOW)

    print("RECONSTRUCTED SHAPE:", reconstructed.shape)
    print("RECON LEN:", len(reconstructed))
    print("LOOKBACK:", LOOKBACK)

    # =====================
    # SLIDING WINDOW
    # =====================
    if len(reconstructed) < LOOKBACK:
        print("❌ DATA KURANG DARI LOOKBACK")
        return {"error": "Data kurang dari lookback"}

    window = reconstructed[-LOOKBACK:]
    window = np.expand_dims(window, axis=0)

    print("WINDOW SHAPE (BEFORE PREDICT):", window.shape)

    # =====================
    # PREDICT
    # =====================
    pred = model.predict(window)

    print("PREDICTION RAW OUTPUT:", pred)

    return {"prediction": float(pred[0][0])}


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="192.168.1.10",
        port=8000,
        reload=True
    )