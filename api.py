# =========================================================
# API FASTAPI - MSSA-LSTM QoS PREDICTION
# Input: batch List[List[float]] dari Flutter SQLite
# =========================================================

import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# =========================================================
# KONSTANTA
# =========================================================
MIN_ROWS   = 111   # lookback=110, minimal 111 baris
N_FEATURES = 4     # Throughput, Delay, Jitter, SINR

# =========================================================
# STATE
# =========================================================
_model_ready   = False
_model_loading = False

# =========================================================
# BACKGROUND LOADING
# =========================================================
async def _load_model_background():
    global _model_ready, _model_loading
    _model_loading = True
    try:
        print("[API] Loading model...")
        from pipeline import warmup
        await asyncio.to_thread(warmup)
        _model_ready   = True
        _model_loading = False
        print("[API] Model siap")
    except Exception as e:
        _model_loading = False
        print(f"[API] Gagal load model: {e}")
        traceback.print_exc()

# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_model_background())
    yield

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="MSSA-LSTM QoS API",
    lifespan=lifespan
)

# =========================================================
# SCHEMA
# =========================================================
class QoSInput(BaseModel):
    input: list[list[float]]

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        if len(v) == 0:
            raise ValueError("Input kosong")
        for i, row in enumerate(v):
            if len(row) != N_FEATURES:
                raise ValueError(
                    f"Baris ke-{i} harus {N_FEATURES} kolom "
                    f"[Throughput, Delay, Jitter, SINR], "
                    f"diterima {len(row)}"
                )
        return v

# =========================================================
# HELPER
# =========================================================
def _check_model() -> Optional[JSONResponse]:
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content={
                "status"       : "loading",
                "message"      : "Model masih loading, coba beberapa saat lagi",
                "model_ready"  : _model_ready,
                "model_loading": _model_loading,
            }
        )
    return None

def _check_min_rows(n: int) -> Optional[JSONResponse]:
    if n < MIN_ROWS:
        return JSONResponse(
            status_code=202,
            content={
                "status" : "waiting",
                "message": f"Data belum cukup: {n}/{MIN_ROWS} baris",
            }
        )
    return None

# =========================================================
# ROOT & STATUS
# =========================================================
@app.get("/")
async def root():
    return {
        "status"       : "online",
        "model"        : "MSSA-LSTM",
        "model_ready"  : _model_ready,
        "model_loading": _model_loading,
    }

@app.get("/status")
async def status():
    return {
        "status"       : "online",
        "model_ready"  : _model_ready,
        "model_loading": _model_loading,
    }

# =========================================================
# /predict — 1 prediksi dari window terakhir
# =========================================================
@app.post("/predict")
async def predict(data: QoSInput):

    err = _check_model()
    if err: return err

    err = _check_min_rows(len(data.input))
    if err: return err

    try:
        from pipeline import predict_current

        result = await asyncio.to_thread(
            predict_current,
            data.input
        )

        pred = result["current_prediction"]

        return {
            "status": "completed",
            "model" : "MSSA-LSTM",
            "result": {
                "final_prediction": pred["qos_index"],
                "series"          : [pred["qos_index"]],
                "forecast_time"   : "t+1",
                "model"           : "MSSA-LSTM",
                "detail"          : {
                    "Throughput (Mbps)": pred["Throughput (Mbps)"],
                    "Delay (ms)"       : pred["Delay (ms)"],
                    "Jitter (ms)"      : pred["Jitter (ms)"],
                    "SINR (dB)"        : pred["SINR (dB)"],
                    "qos_index"        : pred["qos_index"],
                }
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e),
                     "trace": traceback.format_exc()}
        )

# =========================================================
# /predict_sliding — semua titik prediksi sliding window
# =========================================================
@app.post("/predict_sliding")
async def predict_sliding(data: QoSInput):

    err = _check_model()
    if err: return err

    err = _check_min_rows(len(data.input))
    if err: return err

    try:
        from pipeline import predict_sliding as run_sliding

        result = await asyncio.to_thread(
            run_sliding,
            data.input
        )

        return {
            "status"      : "success",
            "model"       : "MSSA-LSTM",
            "total_points": result["total_points"],
            "predictions" : [
                p["qos_index"] for p in result["predictions"]
            ],
            "predictions_detail": result["predictions"],
        }

    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e),
                     "trace": traceback.format_exc()}
        )

# =========================================================
# /predict_future — replika otomatis ke depan
#
# PERUBAHAN DARI VERSI LAMA:
#   - target_steps TIDAK lagi hardcode = 8
#   - Dihitung otomatis di pipeline dari jumlah data
#   - Flutter TIDAK perlu kirim target_steps
#   - Makin banyak data → makin banyak prediksi yang dihasilkan
#
# Rumus step:
#   extra_detik  = (N - lookback) * 1 detik/baris
#   target_steps = extra_detik // 1800
#   (minimal 1 step)
#
# Contoh:
#   N = 1910 baris → 1 prediksi  (t+30m)
#   N = 3710 baris → 2 prediksi  (t+30m, t+60m)
#   N = 5510 baris → 3 prediksi  (dst.)
#
# Response Flutter parse:
#   decoded['status'] == 'success'
#   decoded['predictions']  → List<double> qos_index
#   decoded['total_steps']  → berapa step yang dihasilkan (dinamis)
# =========================================================
@app.post("/predict_future")
async def predict_future(data: QoSInput):

    err = _check_model()
    if err: return err

    err = _check_min_rows(len(data.input))
    if err: return err

    try:
        from pipeline import predict_future as run_future

        # Tidak ada target_steps — pipeline hitung otomatis
        result = await asyncio.to_thread(
            run_future,
            data.input,
        )

        return {
            "status"            : "success",
            "model"             : "MSSA-LSTM",
            "total_steps"       : result["target_steps"],      # dinamis!
            "interval_seconds"  : 1800,
            "forecast_times"    : result["forecast_times"],
            "predictions"       : [
                p["qos_index"] for p in result["future_predictions"]
            ],
            "final_prediction"  : result["final_prediction"],
            "predictions_detail": result["future_predictions"],
        }

    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e),
                     "trace": traceback.format_exc()}
        )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )