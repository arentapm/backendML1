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
MIN_ROWS   = 110
N_FEATURES = 4

JOB_TTL_SECONDS = 6 * 3600

# =========================================================
# STATE
# =========================================================
_model_ready   = False
_model_loading = False

_jobs: dict[str, dict] = {}

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
    now     = time.time()
    expired = [
        jid for jid, j in _jobs.items()
        if now - j["created_at"] > JOB_TTL_SECONDS
    ]
    for jid in expired:
        _jobs.pop(jid, None)


# =========================================================
# LIFESPAN — await warmup agar server online = model ready
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _load_model_background()   # tunggu sampai model benar-benar siap
    yield
    _executor.shutdown(wait=False)


# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="MSSA-LSTM QoS API",
    description=(
        "Recursive Sliding Window Forecasting. "
        "mode=5m → 300 iterasi (cepat). "
        "mode=30m → 7200 iterasi (2 jam)."
    ),
    lifespan=lifespan,
)

# =========================================================
# SCHEMA
# =========================================================
class QoSInput(BaseModel):
    input: list[list[float]]
    mode: str = "30m"   # "5m" atau "30m"

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

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in ("5m", "30m"):
            raise ValueError("mode harus '5m' atau '30m'")
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
# /predict — 1 prediksi real-time
# =========================================================
@app.post("/predict")
async def predict(data: QoSInput):
    err = _check_model()
    if err: return err

    err = _check_min_rows(len(data.input))
    if err: return err

    try:
        from pipeline import predict_current

        result = await asyncio.to_thread(predict_current, data.input)
        pred   = result["current_prediction"]

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
# WORKER — dijalankan di thread terpisah
# =========================================================
def _run_predict_future_job(job_id: str, input_data: list[list[float]], mode: str):
    from pipeline import predict_future as run_future

    def on_progress(done: int, total: int):
        if job_id in _jobs:
            _jobs[job_id]["progress"] = round(done / total * 100, 1)

    try:
        _jobs[job_id]["status"] = "processing"

        result = run_future(input_data, mode=mode, progress_callback=on_progress)

        _jobs[job_id].update({
            "status"  : "success",
            "progress": 100,
            "result"  : {
                "predictions_5m_detail": result["predictions_5m_detail"],
                "predictions_30m"      : result["predictions_30m"],
            },
            "message": None,
        })

    except ValueError as e:
        _jobs[job_id].update({"status": "error", "message": str(e)})
    except Exception as e:
        traceback.print_exc()
        _jobs[job_id].update({"status": "error", "message": str(e)})


# =========================================================
# /predict_future — START JOB
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
        "mode"      : data.mode,
        "created_at": time.time(),
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_predict_future_job,
        job_id,
        data.input,
        data.mode,
    )

    return {
        "status" : "queued",
        "job_id" : job_id,
        "mode"   : data.mode,
        "message": f"Forecast mode={data.mode} sedang diproses, gunakan job_id untuk polling",
    }


# =========================================================
# /predict_future/{job_id} — POLLING STATUS
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
            "status"               : "success",
            "progress"             : 100,
            "model"                : "MSSA-LSTM",
            "mode"                 : job.get("mode", "30m"),
            "predictions_5m_detail": job["result"]["predictions_5m_detail"],
            "predictions_30m"      : job["result"]["predictions_30m"],
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
        "mode"    : job.get("mode", "30m"),
    }


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)