# =========================================================
# API FASTAPI - MSSA-LSTM QoS PREDICTION
# Input: batch List[List[float]] dari Flutter SQLite
# Referensi: Recursive Sliding Window (arXiv:2511.11630)
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
MIN_ROWS   = 110   # = lookback (tidak perlu +1 karena predict_future
                   #   tidak butuh data SETELAH window, window IS-nya data)
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
    description=(
        "Recursive Sliding Window Forecasting, "
        "setiap prediksi menjadi input "
        "window berikutnya (replika otomatis)."
    ),
    lifespan=lifespan,
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
# /predict — 1 prediksi dari window terakhir (real-time)
#
# Dipakai untuk monitoring langsung:
#   - Ambil N baris terakhir dari buffer SQLite
#   - Prediksi 1 step ke depan tanpa rekursi
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
            data.input,
        )

        pred = result["current_prediction"]

        return {
            "status": "success",
            "model" : "MSSA-LSTM",
            "result": {
                "final_prediction" : pred["qos_index"],
                "series"           : [pred["qos_index"]],
                "forecast_time"    : "t+1",
                "model"            : "MSSA-LSTM",
                "detail"           : {
                    "Throughput (Mbps)": pred["Throughput (Mbps)"],
                    "Delay (ms)"       : pred["Delay (ms)"],
                    "Jitter (ms)"      : pred["Jitter (ms)"],
                    "SINR (dB)"        : pred["SINR (dB)"],
                    "qos_index"        : pred["qos_index"],
                },
            },
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status" : "error",
                "message": str(e),
                "trace"  : traceback.format_exc(),
            }
        )

# =========================================================
# /predict_future — Recursive Sliding Window (arXiv:2511.11630)
#
# MEKANISME (identik Gambar 2 jurnal):
#   Step 1 : window = [t1, t2, …, t110]           → P1
#   Step 2 : window = [t2, t3, …, t110, P1]       → P2
#   Step 3 : window = [t3, t4, …, t110, P1, P2]   → P3
#   …
#   Baris tertua dibuang, prediksi terbaru (replika) masuk di ekor.
#
# JUMLAH STEP dihitung OTOMATIS dari jumlah baris data (N):
#   extra_detik  = (N - lookback) * 1 detik/baris
#   target_steps = max(1, extra_detik // 1800)
#
#   Contoh:
#     N = 110  → steps = 1  (minimal, prediksi t+30m)
#     N = 1910 → steps = 1  (t+30m)
#     N = 3710 → steps = 2  (t+30m, t+60m)
#     N = 5510 → steps = 3  (dst.)
#
# Flutter parse response:
#   decoded['status']           == 'success'
#   decoded['predictions']      → List<double> qos_index per step
#   decoded['total_steps']      → int, berapa step dihasilkan
#   decoded['forecast_times']   → List<String> ["t+30m", "t+60m", ...]
#   decoded['predictions_detail'] → List detail tiap step
# =========================================================
@app.post("/predict_future")
async def predict_future(data: QoSInput):

    err = _check_model()
    if err: return err

    err = _check_min_rows(len(data.input))
    if err: return err

    try:
        from pipeline import predict_future as run_future

        result = await asyncio.to_thread(
            run_future,
            data.input,
        )

        return {
            "status"            : "success",
            "model"             : "MSSA-LSTM",
            "total_steps"       : result["target_steps"],
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
            content={
                "status" : "error",
                "message": str(e),
                "trace"  : traceback.format_exc(),
            }
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