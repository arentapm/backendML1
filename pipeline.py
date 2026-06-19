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
BASE_DIR             = Path(__file__).resolve().parent
MODELS_DIR           = BASE_DIR / "models"
BUFFER_SIZE          = 1800   # 30 menit data (1 baris = 1 detik)
PREDICT_INTERVAL     = 1
MIN_DATA_TO_PRED     = 110    # = lookback
DATA_INTERVAL_SECONDS   = 1
FORECAST_INTERVAL_SECONDS = 1800  # 30 menit

# =========================================================
# ARTIFACTS
# =========================================================
_model        = None
_scaler_feat  = None
_config       = None
_buffer       = deque(maxlen=BUFFER_SIZE)
_tick_count   = 0
_last_result  = None


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
# =========================================================
def _predict_one_window(window_scaled: np.ndarray) -> dict:
    """
    window_scaled : shape (lookback, 4) — sudah dinormalisasi & direkonstruksi MSSA
    Returns       : dict prediksi lengkap (skala asli)
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
# HITUNG TARGET_STEPS OTOMATIS
#
# Logika (sesuai jurnal "Predicting Grain Growth..." arXiv:2511.11630):
#   - Seed window = lookback baris pertama (data asli)
#   - Setiap step menggeser window 1 prediksi ke depan
#   - Jumlah step = berapa kali 30 menit muat di sisa data
#
# Contoh:
#   N = 110  → extra=0  → steps=1  (minimal 1 prediksi)
#   N = 1910 → extra=1800s → steps=1
#   N = 3710 → extra=3600s → steps=2
# =========================================================
def _compute_target_steps(
    n_rows:              int,
    lookback:            int,
    data_interval_sec:   int = DATA_INTERVAL_SECONDS,
    forecast_interval_sec: int = FORECAST_INTERVAL_SECONDS,
) -> int:
    extra_seconds = (n_rows - lookback) * data_interval_sec
    steps         = max(1, int(extra_seconds // forecast_interval_sec))
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
    Urutan kolom: [Throughput, Delay, Jitter, SINR]
    Returns   : {"current_prediction": {...}}
    """
    if _model is None:
        warmup()

    lookback = _config["lookback"]
    L_MSSA   = _config["L_MSSA"]

    data_raw   = np.array(raw_input, dtype=float)
    data_raw   = pd.DataFrame(data_raw).ffill().bfill().values
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
# CONFIG — tambahkan ini di bagian atas
# =========================================================
BASE_DIR              = Path(__file__).resolve().parent
MODELS_DIR            = BASE_DIR / "models"
BUFFER_SIZE           = 1800
PREDICT_INTERVAL      = 1
MIN_DATA_TO_PRED      = 110
DATA_INTERVAL_SECONDS = 1

MAX_HOURS             = 12                        # tampilan maksimal 12 jam
INTERVAL_5M_SEC       = 5  * 60                  # 300 detik per titik
INTERVAL_30M_SEC      = 30 * 60                  # 1800 detik per titik
TAMPILAN_5M           = (MAX_HOURS * 60) // 5    # 144 titik
TAMPILAN_30M          = (MAX_HOURS * 60) // 30   # 24 titik
N_FUTURE_TOTAL        = TAMPILAN_5M * INTERVAL_5M_SEC  # 43.200 step (sama untuk keduanya)


# =========================================================
# PREDICT FUTURE — Recursive Sliding Window
#
# Input  : tepat 110 baris data nyata (lookback)
# Loop   : 43.200 kali — persis seperti kode dosen (n_future=300)
#          tapi n_future-nya 43.200 karena mau prediksi 12 jam
#
# Hasil diambil dari:
#   - Setiap step ke-300, 600, ..., 43200  → tampilan per 5 menit (144 titik)
#   - Setiap step ke-1800, 3600, ..., 43200 → tampilan per 30 menit (24 titik)
#
# Contoh analogi kode dosen:
#   dosen: n_future=300 → ambil semua 300 hasil
#   kita : n_future=43200 → ambil hasil[299], hasil[599], ... (per interval)
# =========================================================
def predict_future(raw_input: list[list[float]]) -> dict:
    """
    raw_input : list of list, shape TEPAT (lookback=110, 4)
                Urutan kolom: [Throughput, Delay, Jitter, SINR]
    Returns:
    {
        "predictions_5m"  : [ { "label": "t+5m",  "Throughput": ..., "qos_index": ... }, ... ],  # 144 titik
        "predictions_30m" : [ { "label": "t+30m", "Throughput": ..., "qos_index": ... }, ... ],  # 24 titik
    }
    """
    if _model is None:
        warmup()

    lookback = _config["lookback"]   # 110
    L_MSSA   = _config["L_MSSA"]

    # ── Validasi input ──────────────────────────────────────────────
    data_raw = np.array(raw_input, dtype=float)
    if data_raw.shape[0] != lookback:
        raise ValueError(
            f"Input harus tepat {lookback} baris, "
            f"diterima {data_raw.shape[0]} baris."
        )

    # ── Pra-proses: isi NaN → scale → MSSA (1x di awal saja) ───────
    data_raw      = pd.DataFrame(data_raw).ffill().bfill().values
    data_scaled   = _scaler_feat.transform(data_raw)
    reconstructed = _mssa_reconstruct(data_scaled, L=L_MSSA)   # (110, 4)

    # ── Seed window — titik awal rekursi ────────────────────────────
    window = reconstructed.copy()   # shape (110, 4)

    # ── Tampung SEMUA hasil mentah ───────────────────────────────────
    # Kita simpan semua 43.200 prediksi dulu, lalu slice per interval.
    # Ini persis pola kode dosen — loop n_future kali, append hasilnya.
    all_preds = []   # list of dict, panjang = N_FUTURE_TOTAL

    for step in range(N_FUTURE_TOTAL):

        # Langkah A — prediksi 1 detik ke depan dari window saat ini
        pred = _predict_one_window(window)   # dict: Throughput, Delay, Jitter, SINR, qos_index
        all_preds.append(pred)

        # Langkah B — pred (skala asli) → skala normal = "replika"
        pred_arr    = np.array([[
            pred["Throughput (Mbps)"],
            pred["Delay (ms)"],
            pred["Jitter (ms)"],
            pred["SINR (dB)"],
        ]])
        pred_scaled = _scaler_feat.transform(pred_arr)   # (1, 4)

        # Langkah C — geser window: buang tertua, sisipkan replika di ekor
        #   Sebelum: [r2, r3, ..., r110]
        #   Sesudah: [r3, r4, ..., r110, P_k]
        window = np.vstack([window[1:], pred_scaled])    # tetap (110, 4)

    # ── Slice hasil per interval tampilan ───────────────────────────
    #
    # Interval 5 menit = ambil step ke-300, 600, 900, ..., 43200
    # Index Python    = 299, 599, 899, ..., 43199
    #
    # Interval 30 menit = ambil step ke-1800, 3600, ..., 43200
    # Index Python      = 1799, 3599, ..., 43199

    predictions_5m  = []
    predictions_30m = []

    for i in range(TAMPILAN_5M):                    # i = 0..143
        idx   = (i + 1) * INTERVAL_5M_SEC - 1      # 299, 599, ..., 43199
        menit = (i + 1) * 5
        p     = all_preds[idx]
        predictions_5m.append({
            "label"            : f"t+{menit}m",
            "Throughput (Mbps)": p["Throughput (Mbps)"],
            "Delay (ms)"       : p["Delay (ms)"],
            "Jitter (ms)"      : p["Jitter (ms)"],
            "SINR (dB)"        : p["SINR (dB)"],
            "qos_index"        : p["qos_index"],
        })

    for i in range(TAMPILAN_30M):                   # i = 0..23
        idx   = (i + 1) * INTERVAL_30M_SEC - 1     # 1799, 3599, ..., 43199
        menit = (i + 1) * 30
        p     = all_preds[idx]
        predictions_30m.append({
            "label"            : f"t+{menit}m",
            "Throughput (Mbps)": p["Throughput (Mbps)"],
            "Delay (ms)"       : p["Delay (ms)"],
            "Jitter (ms)"      : p["Jitter (ms)"],
            "SINR (dB)"        : p["SINR (dB)"],
            "qos_index"        : p["qos_index"],
        })

    return {
        "predictions_5m" : predictions_5m,   # 144 titik, label t+5m s/d t+720m
        "predictions_30m": predictions_30m,  # 24 titik,  label t+30m s/d t+720m
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

    result      = predict_current(list(_buffer))
    _last_result = {
        "status"     : "ok",
        **result["current_prediction"],
        "buffer_size": len(_buffer),
    }
    return _last_result