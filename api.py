# =========================================================
# API FASTAPI - MSSA-LSTM QoS PREDICTION
# Input: batch List[List[float]] dari Flutter SQLite
# Referensi: Recursive Sliding Window (arXiv:2511.11630)
#
# /predict_future sekarang ASYNC JOB:
#   1. POST /predict_future       → return job_id langsung (instan)
#   2. GET  /predict_future/{id}  → polling status & progress
#   3. Saat status == "success"   → hasil ada di response (predictions_5m/30m)
# =========================================================

import time
import uuid
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# =========================================================
# KONSTANTA
# =========================================================
MIN_ROWS   = 110   # = lookback
N_FEATURES = 4     # Throughput, Delay, Jitter, SINR

JOB_TTL_SECONDS = 6 * 3600   # job lama dihapus otomatis setelah 6 jam

# =========================================================
# STATE
# =========================================================
_model_ready   = False
_model_loading = False

# Job store di memori — aman karena 1 worker/proses (lihat catatan di bawah)
# Struktur tiap job:
# {
#   "status"   : "queued" | "processing" | "success" | "error" | "waiting",
#   "progress" : 0-100,
#   "message"  : str | None,
#   "result"   : dict | None,
#   "created_at": float (epoch),
# }
_jobs: dict[str, dict] = {}

# Thread pool khusus untuk kerja berat (model inference berulang).
# Dipisah dari default executor agar tidak bentrok dengan endpoint lain.
_executor = ThreadPoolExecutor(max_workers=2)


# =========================================================
# BACKGROUND LOADING MODEL
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


def _cleanup_old_jobs():
    now = time.time()
    expired = [
        jid for jid, j in _jobs.items()
        if now - j["created_at"] > JOB_TTL_SECONDS
    ]
    for jid in expired:
        _jobs.pop(jid, None)


# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _load_model_background()
    yield
    _executor.shutdown(wait=False)


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
# /predict — 1 prediksi dari window terakhir (real-time, tetap sinkron)
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
# WORKER — dijalankan di thread terpisah, update _jobs[job_id]
# =========================================================
def _run_predict_future_job(job_id: str, input_data: list[list[float]]):
    from pipeline import predict_future as run_future

    def on_progress(done: int, total: int):
        if job_id in _jobs:
            _jobs[job_id]["progress"] = round(done / total * 100, 1)

    try:
        _jobs[job_id]["status"] = "processing"

        result = run_future(input_data, progress_callback=on_progress)

        _jobs[job_id].update({
            "status"  : "success",
            "progress": 100,
            "result"  : {
                "predictions_5m_detail": result["predictions_5m_detail"], 
                "predictions_30m": result["predictions_30m"],
            },
            "message": None,
        })

    except ValueError as e:
        _jobs[job_id].update({
            "status" : "error",
            "message": str(e),
        })
    except Exception as e:
        traceback.print_exc()
        _jobs[job_id].update({
            "status" : "error",
            "message": str(e),
        })


# =========================================================
# /predict_future — START JOB (instan, tidak menunggu)
#
# Response:
#   {"status": "queued", "job_id": "..."}
# Flutter lalu polling GET /predict_future/{job_id}
# =========================================================
@app.post("/predict_future")
async def predict_future_start(data: QoSInput):

    err = _check_model()
    if err: return err

    err = _check_min_rows(len(data.input))
    if err: return err

    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status"    : "queued",
        "progress"  : 0,
        "message"   : None,
        "result"    : None,
        "created_at": time.time(),
    }

    loop = asyncio.get_running_loop()
    # Lempar kerja berat ke thread pool — tidak diawait, supaya endpoint
    # langsung return job_id ke client.
    loop.run_in_executor(_executor, _run_predict_future_job, job_id, data.input)

    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Forecast sedang diproses, gunakan job_id untuk polling",
    }


# =========================================================
# /predict_future/{job_id} — POLLING STATUS
#
# Response saat masih proses:
#   {"status": "processing", "progress": 42.5}
# Response saat sukses:
#   {"status": "success", "progress": 100,
#    "predictions_5m": [...], "predictions_30m": [...]}
# Response saat gagal:
#   {"status": "error", "message": "..."}
# =========================================================
@app.get("/predict_future/{job_id}")
async def predict_future_status(job_id: str):
    job = _jobs.get(job_id)

    if job is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "job_id tidak ditemukan atau sudah kedaluwarsa"},
        )

    if job["status"] == "success":
        return {
            "status"          : "success",
            "progress"        : 100,
            "model"           : "MSSA-LSTM",
            "predictions_5m_detail": job["result"]["predictions_5m_detail"],
            "predictions_30m" : job["result"]["predictions_30m"],
        }

    if job["status"] == "error":
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": job["message"]},
        )

    # queued / processing
    return {
        "status"  : job["status"],
        "progress": job["progress"],
    }


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