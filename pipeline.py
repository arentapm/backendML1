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
# PREDICT FUTURE — Recursive Sliding Window (jurnal arXiv:2511.11630)
#
# KONSEP (persis seperti di jurnal):
#   Diberikan window awal berisi data nyata [t1 … t_lookback]:
#
#   Step 1:  input = [t1,  t2,  …, t110]          → prediksi P1
#   Step 2:  input = [t2,  t3,  …, t110, P1]      → prediksi P2
#   Step 3:  input = [t3,  t4,  …, t110, P1, P2]  → prediksi P3
#   …
#   Step k:  window tertua dibuang, replika terbaru masuk di ekor
#
#   Ini identik dengan Gambar 2 & Persamaan (3)–(4) di jurnal tsb:
#       y1 = f(x_t0, x_t1, x_t2, x_t3, x_t4)   ← versi jurnal (window=5)
#       y2 = f(x_t1, x_t2, x_t3, x_t4, y1)
#       y3 = f(x_t2, x_t3, x_t4, y1,   y2)
#
#   Pada kode ini window = lookback (110), 
#   prediksi sebelumnya menggantikan elemen tertua → window bergeser maju terus.
#
# Dipanggil oleh /predict_future
# =========================================================
def predict_future(raw_input: list[list[float]]) -> dict:
    """
    raw_input : list of list, shape (N, 4), N >= lookback
    Returns:
    {
        "future_predictions": [
            {
                "Throughput (Mbps)": float,
                "Delay (ms)"       : float,
                "Jitter (ms)"      : float,
                "SINR (dB)"        : float,
                "qos_index"        : float,
                "step"             : int,   # 1-based
            },
            ...
        ],
        "final_prediction" : float,          # qos_index langkah terakhir
        "forecast_times"   : ["t+30m", ...], # label waktu tiap step
        "target_steps"     : int,
    }
    """
    if _model is None:
        warmup()

    lookback = _config["lookback"]
    L_MSSA   = _config["L_MSSA"]

    # ── Pra-proses: isi NaN, scale, rekonstruksi MSSA ──────────────
    data_raw      = np.array(raw_input, dtype=float)
    data_raw      = pd.DataFrame(data_raw).ffill().bfill().values
    data_scaled   = _scaler_feat.transform(data_raw)
    reconstructed = _mssa_reconstruct(data_scaled, L=L_MSSA)

    N = len(reconstructed)
    if N < lookback:
        raise ValueError(f"Data tidak cukup: {N} baris, butuh minimal {lookback}")

    # ── Hitung berapa step yang akan diprediksi ─────────────────────
    target_steps = _compute_target_steps(N, lookback)

    # ── DEBUG: cek seed window ──────────────────────────────────────
    seed_window = reconstructed[-lookback:].copy()   # shape (lookback, 4)
    seed_ori    = _scaler_feat.inverse_transform(seed_window)
    print(f"[DEBUG] Seed window mean  (asli) = {seed_ori.mean(axis=0).round(4)}")
    print(f"[DEBUG] Seed window akhir (asli) = {seed_ori[-1].round(4)}")

    # ── RECURSIVE SLIDING WINDOW (inti jurnal) ──────────────────────
    #
    #  Iterasi 1  : window = [t1 … t_lookback]           → P1
    #  Iterasi 2  : window = [t2 … t_lookback, P1_sc]    → P2
    #  Iterasi k  : window = [..., P(k-2)_sc, P(k-1)_sc] → Pk
    #
    #  Setiap hasil prediksi (skala asli) di-transform balik ke
    #  skala normal (pred_scaled) lalu di-vstack ke ekor window,
    #  sementara baris paling lama (index 0) dibuang.
    # ───────────────────────────────────────────────────────────────
    future_preds = []
    window       = seed_window.copy()   # shape (lookback, 4), skala normal

    for step in range(target_steps):

        # Langkah A — prediksi 1 step ke depan
        pred = _predict_one_window(window)
        pred["step"] = step + 1
        future_preds.append(pred)

        print(
            f"[DEBUG] Step {step+1:02d}: "
            f"Throughput={pred['Throughput (Mbps)']:.4f}  "
            f"Delay={pred['Delay (ms)']:.4f}  "
            f"Jitter={pred['Jitter (ms)']:.4f}  "
            f"SINR={pred['SINR (dB)']:.4f}  "
            f"→ QoS={pred['qos_index']:.4f}"
        )

        # Langkah B — konversi prediksi (skala asli) → skala normal
        #             ini yang disebut "replika"
        pred_arr    = np.array([[
            pred["Throughput (Mbps)"],
            pred["Delay (ms)"],
            pred["Jitter (ms)"],
            pred["SINR (dB)"],
        ]])
        pred_scaled = _scaler_feat.transform(pred_arr)   # shape (1, 4)

        # Langkah C — geser window: buang baris tertua, sisipkan replika
        #
        #   Sebelum : [t1,  t2,  …, t(lookback)]
        #   Sesudah  : [t2,  t3,  …, t(lookback), Pk_scaled]
        window = np.vstack([window[1:], pred_scaled])   # shape tetap (lookback, 4)

    return {
        "future_predictions": future_preds,
        "final_prediction"  : future_preds[-1]["qos_index"],
        "forecast_times"    : [
            f"t+{(i + 1) * (FORECAST_INTERVAL_SECONDS // 60)}m"
            for i in range(target_steps)
        ],
        "target_steps"      : target_steps,
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