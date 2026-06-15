import json
import joblib
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from scipy.linalg import svd
from tensorflow.keras.models import load_model

# =========================================================
# CONFIG
# =========================================================
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

BUFFER_SIZE      = 1800   # 30 menit data (1 baris = 1 detik)
PREDICT_INTERVAL = 1
MIN_DATA_TO_PRED = 110    # = lookback

# Interval: 1 baris data = 1 detik, 1 step forecast = 30 menit
DATA_INTERVAL_SECONDS     = 1
FORECAST_INTERVAL_SECONDS = 1800   # 30 menit

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
    _model       = load_model(MODELS_DIR / "model_qos_dengan_MSSA.keras")
    _scaler_feat = joblib.load(MODELS_DIR / "scaler_feat.pkl")
    with open(MODELS_DIR / "config.json") as f:
        _config = json.load(f)
    print(
        f"[Backend] Model loaded | "
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
    if avg >= 3.8:   qos = 95 + ((avg - 3.8) / 0.2) * 5
    elif avg >= 3.0: qos = 75 + ((avg - 3.0) / 0.79) * 19.75
    elif avg >= 2.0: qos = 50 + ((avg - 2.0) / 0.99) * 24.75
    else:            qos = 25 + ((avg - 1.0) / 0.99) * 24.75
    return float(np.clip(qos, 25, 100))


# =========================================================
# MSSA
# =========================================================
def _mssa_reconstruct(data_scaled, L=50):
    X    = np.array(data_scaled, dtype=float)
    N, M = X.shape
    L    = min(L, N // 3)
    K    = N - L + 1

    trajectory = np.zeros((L * M, K))
    for m in range(M):
        s = X[:, m]
        for i in range(K):
            trajectory[m * L:(m + 1) * L, i] = s[i:i + L]

    U, S, Vt = svd(trajectory, full_matrices=False)

    n_trend = 1
    n_fluct = 19

    def reconstruct(indices):
        out = np.zeros((N, M))
        for m in range(M):
            br = slice(m * L, (m + 1) * L)
            Xr = np.zeros((L, K))
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
        reconstruct(list(range(n_trend)))
        + reconstruct(list(range(n_trend, n_trend + n_fluct)))
    )


# =========================================================
# HELPER: 1 PREDIKSI DARI 1 WINDOW
# =========================================================
def _predict_one_window(window_scaled: np.ndarray) -> dict:
    """
    window_scaled : shape (lookback, 4) — sudah dinormalisasi & direkonstruksi MSSA
    Returns       : dict prediksi lengkap
    """
    lookback = _config["lookback"]
    inp      = window_scaled[-lookback:].astype(np.float32).reshape(1, lookback, 4)
    pred_sc  = _model.predict(inp, verbose=0)[0]
    pred_ori = _scaler_feat.inverse_transform(pred_sc.reshape(1, -1))[0]
    qos      = compute_qos_index(*pred_ori)
    return {
        "Throughput (Mbps)": round(float(pred_ori[0]), 4),
        "Delay (ms)"       : round(float(pred_ori[1]), 4),
        "Jitter (ms)"      : round(float(pred_ori[2]), 4),
        "SINR (dB)"        : round(float(pred_ori[3]), 4),
        "qos_index"        : round(qos, 4),
    }


# =========================================================
# HITUNG TARGET_STEPS OTOMATIS — inti "replika otomatis"
#
# Logika:
#   - Kita punya N baris data, interval 1 detik
#   - lookback = 110 baris dipakai sebagai seed window
#   - Sisa data = (N - lookback) baris = (N - lookback) detik
#   - 1 step forecast = 1800 detik (30 menit)
#   - target_steps = sisa_detik // 1800
#
# Contoh:
#   N = 1910 → sisa = 1800 detik → 1 step  (prediksi t+30m)
#   N = 3710 → sisa = 3600 detik → 2 step  (prediksi t+30m, t+60m)
#   N = 5510 → sisa = 5400 detik → 3 step  (dst.)
#
# Kalau data sedikit (misal N = 200):
#   sisa = 90 detik → 0 step → pakai minimum 1 step
# =========================================================
def _compute_target_steps(
    n_rows: int,
    lookback: int,
    data_interval_sec: int    = DATA_INTERVAL_SECONDS,
    forecast_interval_sec: int = FORECAST_INTERVAL_SECONDS,
) -> int:
    extra_seconds = (n_rows - lookback) * data_interval_sec
    steps         = extra_seconds // forecast_interval_sec
    steps         = max(1, int(steps))   # minimal selalu 1 prediksi
    print(
        f"[Replika Otomatis] N={n_rows}, lookback={lookback}, "
        f"extra={extra_seconds}s → target_steps={steps}"
    )
    return steps


# =========================================================
# PREDICT CURRENT — 1 prediksi dari window terakhir
# Dipanggil oleh /predict
# =========================================================
def predict_current(raw_input: list[list[float]]) -> dict:
    """
    raw_input : list of list, shape (N, 4), N >= lookback
                urutan kolom: [Throughput, Delay, Jitter, SINR]
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
# PREDICT SLIDING — semua titik prediksi sliding window
# Dipanggil oleh /predict_sliding
# =========================================================
def predict_sliding(raw_input: list[list[float]]) -> dict:
    """
    raw_input : list of list, shape (N, 4), N >= lookback+1
    Returns   : {
        "total_points": int,
        "predictions" : [{"Throughput":..., ..., "qos_index":..., "step": i}]
    }
    """
    if _model is None:
        warmup()

    lookback = _config["lookback"]
    L_MSSA   = _config["L_MSSA"]

    data_raw    = np.array(raw_input, dtype=float)
    data_raw    = pd.DataFrame(data_raw).ffill().bfill().values
    data_scaled = _scaler_feat.transform(data_raw)

    reconstructed = _mssa_reconstruct(data_scaled, L=L_MSSA)

    N = len(reconstructed)
    if N < lookback + 1:
        raise ValueError(
            f"Data tidak cukup: {N} baris, "
            f"butuh minimal {lookback + 1}"
        )

    total_points = N - lookback
    predictions  = []

    for i in range(total_points):
        window = reconstructed[i: i + lookback]
        pred   = _predict_one_window(window)
        pred["step"] = i
        predictions.append(pred)

    return {
        "total_points": total_points,
        "predictions" : predictions,
    }


# =========================================================
# PREDICT FUTURE — replika otomatis ke depan
# Dipanggil oleh /predict_future
#
# KONSEP REPLIKA OTOMATIS:
#   1. Hitung target_steps dari jumlah data (TIDAK hardcode)
#   2. Ambil window terakhir dari data asli sebagai seed
#   3. Loop per step:
#        a. Prediksi 1 step ke depan dari window saat ini
#        b. Hasil prediksi di-scale → masuk ke window (REPLIKA)
#        c. Window digeser: buang baris tertua, tambah replika
#   4. Ulangi sampai target_steps terpenuhi
#
# Kenapa hasilnya TIDAK flat/identik?
#   Karena LSTM non-linear: input sedikit beda → output beda.
#   Namun makin banyak replika masuk window, uncertainty naik
#   dan model cenderung konvergen ke nilai equilibrium-nya.
#   Ini WAJAR dalam forecasting (fenomena mean reversion).
# =========================================================
def predict_future(raw_input: list[list[float]]) -> dict:
    """
    raw_input    : list of list, shape (N, 4), N >= lookback
    target_steps : dihitung OTOMATIS dari jumlah baris data
                   tidak perlu dikirim dari Flutter

    Returns:
    {
        "future_predictions": [
            {"Throughput (Mbps)":..., "Delay (ms)":...,
             "Jitter (ms)":..., "SINR (dB)":...,
             "qos_index":..., "step": 1},
            ...
        ],
        "final_prediction"  : float,
        "forecast_times"    : ["t+30m", "t+60m", ...],
        "target_steps"      : int,
    }
    """
    if _model is None:
        warmup()

    lookback = _config["lookback"]
    L_MSSA   = _config["L_MSSA"]

    data_raw      = np.array(raw_input, dtype=float)
    data_raw      = pd.DataFrame(data_raw).ffill().bfill().values
    data_scaled   = _scaler_feat.transform(data_raw)
    reconstructed = _mssa_reconstruct(data_scaled, L=L_MSSA)

    N = len(reconstructed)
    if N < lookback:
        raise ValueError(f"Data tidak cukup: {N} < {lookback}")

    # ── REPLIKA OTOMATIS: hitung step dari jumlah data ───
    target_steps = _compute_target_steps(N, lookback)

    future_preds = []

    # Seed window: lookback baris terakhir dari data ASLI
    window = reconstructed[-lookback:].copy()   # shape (lookback, 4)

    # DEBUG: cek nilai seed window sebelum prediksi pertama
    seed_mean = window.mean(axis=0)
    seed_std  = window.std(axis=0)
    print(f"[DEBUG] Seed window (scaled) mean={seed_mean.round(4)} std={seed_std.round(4)}")

    # Inverse seed untuk lihat nilai aslinya
    seed_ori = _scaler_feat.inverse_transform(window)
    print(f"[DEBUG] Seed window (original) mean={seed_ori.mean(axis=0).round(4)}")
    print(f"[DEBUG] Seed window (original) last row={seed_ori[-1].round(4)}")

    for step in range(target_steps):

        # 1. Prediksi 1 step ke depan dari window saat ini
        pred         = _predict_one_window(window)
        pred["step"] = step + 1
        future_preds.append(pred)

        # DEBUG: print tiap step
        print(
            f"[DEBUG] Step {step+1:02d}: "
            f"Throughput={pred['Throughput (Mbps)']:.4f} "
            f"Delay={pred['Delay (ms)']:.4f} "
            f"Jitter={pred['Jitter (ms)']:.4f} "
            f"SINR={pred['SINR (dB)']:.4f} "
            f"→ qos={pred['qos_index']:.4f}"
        )

        # 2. REPLIKA: ubah hasil prediksi (skala asli) → skala normal
        pred_arr = np.array([[
            pred["Throughput (Mbps)"],
            pred["Delay (ms)"],
            pred["Jitter (ms)"],
            pred["SINR (dB)"],
        ]])
        pred_scaled = _scaler_feat.transform(pred_arr)   # shape (1, 4)

        # 3. SLIDING WINDOW: buang baris tertua, tambah replika di akhir
        #    Sebelum: [t1, t2, ..., t110]
        #    Sesudah: [t2, t3, ..., t110, replika_step]
        window = np.vstack([window[1:], pred_scaled])    # shape tetap (lookback, 4)

    return {
        "future_predictions": future_preds,
        "final_prediction"  : future_preds[-1]["qos_index"],
        "forecast_times"    : [
            f"t+{(i + 1) * (FORECAST_INTERVAL_SECONDS // 60)}m"
            for i in range(target_steps)
        ],
        "target_steps": target_steps,
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