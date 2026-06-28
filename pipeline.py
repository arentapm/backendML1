import json
import joblib
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from scipy.linalg import svd
from tensorflow.keras.models import load_model
from typing import Callable, Optional

# =========================================================
# CONFIG
# =========================================================
BASE_DIR              = Path(__file__).resolve().parent
MODELS_DIR            = BASE_DIR / "models"
BUFFER_SIZE           = 1800   # 30 menit data (1 baris = 1 detik)
PREDICT_INTERVAL      = 1
MIN_DATA_TO_PRED      = 110    # = lookback
DATA_INTERVAL_SECONDS = 1

INTERVAL_5M_SEC    = 5  * 60        # 300 detik
INTERVAL_30M_SEC   = 30 * 60        # 1800 detik
TAMPILAN_5M_DETAIL = 300            # semua step t+1..t+300 untuk grafik
TAMPILAN_30M       = (2 * 60) // 30 # 4 titik (t+30m, t+60m, t+90m, t+120m)

# =========================================================
# ARTIFACTS
# =========================================================
_model       = None
_scaler_feat = None
_config      = None
_buffer      = deque(maxlen=BUFFER_SIZE)
_tick_count  = 0
_last_result = None


def warmup():
    global _model, _scaler_feat, _config
    _model       = load_model(MODELS_DIR / "keras_saved_model")
    _scaler_feat = joblib.load(MODELS_DIR / "scaler_feat.pkl")
    with open(MODELS_DIR / "config.json") as f:
        _config = json.load(f)
    print(
        f"[Backend] Keras Model loaded | "
        f"lookback={_config['lookback']} | "
        f"L_MSSA={_config['L_MSSA']}"
    )


# =========================================================
# TIPHON SCORING
# =========================================================
def _score_throughput(x):
    if x >= 75: return 4
    elif x >= 50: return 3
    elif x >= 25: return 2
    return 1

def _score_delay(x):
    if x < 150: return 4
    elif x < 300: return 3
    elif x < 450: return 2
    return 1

def _score_jitter(x):
    if x == 0: return 4
    elif x < 75: return 3
    elif x <= 125: return 2
    return 1

def _score_sinr(x):
    if x > 20: return 4
    elif x >= 15: return 3
    elif x >= 0: return 2
    return 1

def compute_qos_index(throughput, delay, jitter, sinr):
    scores = [
        _score_throughput(throughput),
        _score_delay(delay),
        _score_jitter(jitter),
        _score_sinr(sinr),
    ]
    avg = np.mean(scores)
    if avg >= 3.8:
        qos = 95 + ((avg - 3.8) / 0.2) * 5
    elif avg >= 3.0:
        qos = 75 + ((avg - 3.0) / 0.79) * 19.75
    elif avg >= 2.0:
        qos = 50 + ((avg - 2.0) / 0.99) * 24.75
    else:
        qos = 25 + ((avg - 1.0) / 0.99) * 24.75
    return float(np.clip(qos, 25, 100))


# =========================================================
# MSSA
# =========================================================
def _mssa_reconstruct(data_scaled, L=50):
    X          = np.array(data_scaled, dtype=float)
    N, M       = X.shape
    L          = min(L, N // 3)
    K          = N - L + 1
    trajectory = np.zeros((L * M, K))
    for m in range(M):
        s = X[:, m]
        for i in range(K):
            trajectory[m * L:(m + 1) * L, i] = s[i:i + L]
    U, S, Vt = svd(trajectory, full_matrices=False)
    n_trend  = 1
    n_fluct  = 19

    def reconstruct(indices):
        out = np.zeros((N, M))
        for m in range(M):
            br  = slice(m * L, (m + 1) * L)
            Xr  = np.zeros((L, K))
            for idx in indices:
                Xr += S[idx] * (
                    U[br, idx].reshape(L, 1) @ Vt[idx].reshape(1, K)
                )
            rs = np.zeros(N)
            for d in range(-(L - 1), K):
                rs[d + L - 1] = np.diag(Xr, d).mean()
            out[:, m] = rs
        return out

    return (
        reconstruct(list(range(n_trend))) +
        reconstruct(list(range(n_trend, n_trend + n_fluct)))
    )


# =========================================================
# HELPER: 1 PREDIKSI DARI 1 WINDOW
#
# PENTING — pola rekursif yang BENAR:
#   Model menerima window dalam skala SCALED, outputnya (pred_sc)
#   JUGA dalam skala SCALED. Untuk langkah rekursif berikutnya,
#   pakai pred_sc apa adanya (TANPA inverse_transform lalu transform
#   lagi) agar tidak mengakumulasi error tiap step.
#
#   inverse_transform hanya dipakai untuk PELAPORAN (nilai ke user).
# =========================================================
def _predict_one_window(window_scaled: np.ndarray) -> dict:
    lookback = _config["lookback"]
    inp      = window_scaled[-lookback:].astype(np.float32).reshape(1, lookback, 4)
    pred_sc  = _model.predict(inp, verbose=0)[0]
    pred_ori = _scaler_feat.inverse_transform(pred_sc.reshape(1, -1))[0]
    qos      = compute_qos_index(*pred_ori)

    return {
        "scaled"           : pred_sc,
        "Throughput (Mbps)": round(float(pred_ori[0]), 4),
        "Delay (ms)"       : round(float(pred_ori[1]), 4),
        "Jitter (ms)"      : round(float(pred_ori[2]), 4),
        "SINR (dB)"        : round(float(pred_ori[3]), 4),
        "qos_index"        : round(qos, 4),
    }


# =========================================================
# PREDICT CURRENT — 1 prediksi dari window terakhir
# Dipanggil oleh /predict
# =========================================================
def predict_current(raw_input: list[list[float]]) -> dict:
    """
    raw_input : list of list, shape (N, 4), N >= lookback
    Urutan kolom: [Throughput, Delay, Jitter, SINR]
    Returns   : {"current_prediction": {...}}
    """
    if _model is None:
        warmup()

    lookback = _config["lookback"]
    L_MSSA   = _config["L_MSSA"]

    data_raw      = np.array(raw_input, dtype=float)
    data_raw      = pd.DataFrame(data_raw).ffill().bfill().values
    data_scaled   = _scaler_feat.transform(data_raw)
    reconstructed = _mssa_reconstruct(data_scaled, L=L_MSSA)

    if len(reconstructed) < lookback:
        raise ValueError(
            f"Data tidak cukup setelah MSSA: "
            f"{len(reconstructed)} < {lookback}"
        )

    pred = _predict_one_window(reconstructed)
    return {"current_prediction": pred}


# =========================================================
# PREDICT FUTURE — Recursive Sliding Window
#
# mode="5m"  → loop 300 iterasi  → cepat (detik)
# mode="30m" → loop 7200 iterasi → lebih lama (menit)
#
# Hasil:
#   mode="5m"  → predictions_5m_detail (300 titik t+1s..t+300s)
#   mode="30m" → predictions_5m_detail (300 titik) +
#                predictions_30m (4 titik: t+30m..t+120m)
# =========================================================
def predict_future(
    raw_input: list[list[float]],
    mode: str = "30m",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    progress_every: int = 200,
) -> dict:
    if _model is None:
        warmup()

    lookback = _config["lookback"]
    L_MSSA   = _config["L_MSSA"]

    data_raw = np.array(raw_input, dtype=float)
    if data_raw.shape[0] < lookback:
        raise ValueError(f"Input harus minimal {lookback} baris")

    data_raw      = data_raw[-lookback:]
    data_raw      = pd.DataFrame(data_raw).ffill().bfill().values
    data_scaled   = _scaler_feat.transform(data_raw)
    reconstructed = _mssa_reconstruct(data_scaled, L=L_MSSA)

    window = reconstructed.copy()

    # Tentukan target step berdasarkan mode
    if mode == "5m":
        # Hanya 300 iterasi — cepat
        all_targets = list(range(1, INTERVAL_5M_SEC + 1))  # 1..300
    else:
        # 7200 iterasi — grafik 5m + 4 titik 30m (2 jam)
        targets_5m_detail = set(range(1, INTERVAL_5M_SEC + 1))
        targets_30m       = {(i + 1) * INTERVAL_30M_SEC for i in range(TAMPILAN_30M)}
        all_targets       = sorted(targets_5m_detail | targets_30m)

    target_set = set(all_targets)
    max_step   = all_targets[-1]  # 300 untuk "5m", 7200 untuk "30m"

    print(f"[Backend] predict_future mode={mode} | max_step={max_step}")

    results_at: dict[int, dict] = {}
    next_target_idx = 0

    for step in range(1, max_step + 1):
        pred = _predict_one_window(window)

        if step in target_set:
            results_at[step] = pred

        pred_scaled = pred["scaled"].reshape(1, -1)
        window      = np.vstack([window[1:], pred_scaled])

        if progress_callback is not None and (step % progress_every == 0 or step == max_step):
            progress_callback(step, max_step)

    # Output grafik detail 5m: t+1s..t+300s (300 titik) — selalu ada
    predictions_5m_detail = [
        {"label": f"t+{step}s", "qos_index": results_at[step]["qos_index"]}
        for step in range(1, INTERVAL_5M_SEC + 1)
    ]

    # Output 30m: t+30m, t+60m, t+90m, t+120m (4 titik) — hanya mode 30m
    predictions_30m = []
    if mode == "30m":
        for i in range(TAMPILAN_30M):
            s = (i + 1) * INTERVAL_30M_SEC
            p = results_at[s]
            predictions_30m.append({
                "label"            : f"t+{(i + 1) * 30}m",
                "Throughput (Mbps)": p["Throughput (Mbps)"],
                "Delay (ms)"       : p["Delay (ms)"],
                "Jitter (ms)"      : p["Jitter (ms)"],
                "SINR (dB)"        : p["SINR (dB)"],
                "qos_index"        : p["qos_index"],
            })

    return {
        "predictions_5m_detail": predictions_5m_detail,
        "predictions_30m"      : predictions_30m,
    }


# =========================================================
# PUSH DATA — kompatibilitas streaming
# =========================================================
def push_data(row: dict) -> dict | None:
    global _tick_count, _last_result

    if _model is None:
        warmup()

    features = _config["features"]
    row_arr  = [row.get(f, np.nan) for f in features]

    if any(np.isnan(v) for v in row_arr):
        return _last_result

    _buffer.append(row_arr)
    _tick_count += 1

    if len(_buffer) < MIN_DATA_TO_PRED:
        return {
            "status"   : "buffering",
            "buffer"   : len(_buffer),
            "need"     : MIN_DATA_TO_PRED,
            "qos_index": None,
        }

    if _tick_count % PREDICT_INTERVAL != 0:
        return _last_result

    result       = predict_current(list(_buffer))
    _last_result = {
        "status"     : "ok",
        **result["current_prediction"],
        "buffer_size": len(_buffer),
    }
    return _last_result