# =========================================================
# IMPORT
# =========================================================
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional
from scipy.linalg import svd

# =========================================================
# INIT APP
# =========================================================
app = FastAPI()

print("===================================")
print("API SERVER STARTED")
print("===================================")

# =========================================================
# CONFIG
# =========================================================
LOOKBACK = 110
HORIZON = 4
MSSA_WINDOW = 50

# simulasi weekly (misal 2 jam per hari × 7 hari × 3600 detik)
MIN_WEEKLY_DATA = 7 * 2 * 60 * 60  # 50400 data

# =========================================================
# REQUEST SCHEMA
# =========================================================
class QoSInput(BaseModel):
    input: List[List[float]]
    target: Optional[List[float]] = None

class QoSEvalInput(BaseModel):
    input: List[List[float]]
    target: List[float]

# =========================================================
# LOAD MODEL
# =========================================================
model = tf.keras.models.load_model(
    "model_qos_tiphon1.keras",
    compile=False
)

print("MODEL INPUT SHAPE:", model.input_shape)

# =========================================================
# MSSA FUNCTION (HARUS SAMA DENGAN TRAINING)
# =========================================================
def mssa_decompose(series, L=50):

    N, n_features = series.shape
    L = min(L, N // 3)
    K = N - L + 1

    X = np.zeros((L * n_features, K))

    for i in range(K):
        X[:, i] = series[i:i+L].T.flatten('F')

    U, S, Vt = svd(X, full_matrices=False)

    rank = len(S)
    n_trend = max(1, rank // 4)
    n_fluct = max(1, rank // 4)

    trend_comp = U[:, :n_trend] @ np.diag(S[:n_trend]) @ Vt[:n_trend]

    fluct_comp = (
        U[:, n_trend:n_trend+n_fluct]
        @ np.diag(S[n_trend:n_trend+n_fluct])
        @ Vt[n_trend:n_trend+n_fluct]
    )

    def reconstruct(mat):
        recon = np.zeros((N, n_features))

        for f in range(n_features):
            comp = mat[f*L:(f+1)*L]

            for i in range(N):
                vals = []
                for j in range(max(0, i-L+1), min(i+1, K)):
                    vals.append(comp[i-j, j])

                recon[i, f] = np.mean(vals)

        return recon

    trend = reconstruct(trend_comp)
    fluct = reconstruct(fluct_comp)

    return trend + fluct

# =========================================================
# PREPROCESS
# =========================================================
def preprocess(raw):
    raw = pd.DataFrame(raw)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.interpolate(method='linear', limit_direction='both')
    raw = raw.fillna(raw.median())
    return raw.values

# =========================================================
# PREDICT (WEEKLY FORECASTING)
# =========================================================
@app.post("/predict")
def predict(data: QoSInput):

    try:
        print("\n====================================")
        print("WEEKLY PREDICTION REQUEST")
        print("====================================")

        raw = np.array(data.input, dtype=float)

        # VALIDASI
        if len(raw.shape) != 2 or raw.shape[1] != 4:
            return {"status": "error", "message": "Input harus (n,4)"}

        # WEEKLY CHECK
        if len(raw) < LOOKBACK:
            return {
                "status": "error",
                "message": "Data tidak cukup untuk forecasting"
            }

        # CLEANING
        raw = preprocess(raw)

        # MSSA
        reconstructed = mssa_decompose(raw, L=MSSA_WINDOW)

        # AMBIL WINDOW TERAKHIR (representasi histori)
        window = reconstructed[-LOOKBACK:]
        window = np.expand_dims(window, axis=0)

        # PREDIKSI MULTI-STEP
        pred = model(window, training=False).numpy().flatten()

        result = {
            "status": "success",
            "predictions": {
                "30_min": float(pred[0]),
                "60_min": float(pred[1]),
                "90_min": float(pred[2]),
                "120_min": float(pred[3])
            }
        }

        # OPTIONAL EVALUASI
        if data.target is not None:

            y_true = np.array(data.target, dtype=float)

            if len(y_true) == HORIZON:
                rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
                mae  = float(np.mean(np.abs(y_true - pred)))

                result["evaluation"] = {
                    "rmse": rmse,
                    "mae": mae
                }
            else:
                result["warning"] = "Target harus 4 nilai (multi-step)"

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================================================
# EVALUATE (BATCH WEEKLY)
# =========================================================
@app.post("/evaluate")
def evaluate(data: QoSEvalInput):

    try:
        raw = np.array(data.input, dtype=float)
        y_true = np.array(data.target, dtype=float)

        if len(raw.shape) != 2 or raw.shape[1] != 4:
            return {"status": "error", "message": "Input harus (n,4)"}

        raw = preprocess(raw)
        reconstructed = mssa_decompose(raw, L=MSSA_WINDOW)

        X, y = [], []

        for i in range(len(reconstructed) - LOOKBACK - HORIZON):
            X.append(reconstructed[i:i+LOOKBACK])
            y.append(y_true[i+LOOKBACK:i+LOOKBACK+HORIZON])

        X = np.array(X)
        y = np.array(y)

        pred = model.predict(X)

        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        mae  = float(np.mean(np.abs(y - pred))

        )

        # PER HORIZON
        detail = {}
        for i in range(HORIZON):
            rmse_h = float(np.sqrt(np.mean((y[:, i] - pred[:, i]) ** 2)))
            mae_h  = float(np.mean(np.abs(y[:, i] - pred[:, i])))

            detail[f"horizon_{i+1}"] = {
                "rmse": rmse_h,
                "mae": mae_h
            }

        return {
            "status": "success",
            "overall": {
                "rmse": rmse,
                "mae": mae
            },
            "per_horizon": detail,
            "samples": len(y)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"message": "QoS Weekly Forecasting API Running"}

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="192.168.1.14",
        port=8000,
        reload=True
    )