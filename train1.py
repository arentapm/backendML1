# =========================================================
# TRAINING LSTM TANPA MSSA
# Missing value tetap dipertahankan menggunakan interpolasi
# dan TIDAK menghapus baris time series
# =========================================================

# =========================================================
# IMPORT
# =========================================================
import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# TRAIN FUNCTION
# =========================================================
def train_model():

    print("\n==============================")
    print("TRAINING DIMULAI")
    print("==============================")

    # =====================================================
    # PATH
    # =====================================================
    BASE_DIR = Path(__file__).resolve().parent

    dataset_path = BASE_DIR / "dataset" / "datasettt.csv"

    # SAVE KE models1
    models_dir = BASE_DIR / "models1"
    models_dir.mkdir(exist_ok=True)

    # =====================================================
    # LOAD DATA
    # =====================================================
    df = pd.read_csv(dataset_path)

    # =====================================================
    # NORMALISASI NAMA KOLOM
    # =====================================================
    df.columns = df.columns.str.strip()

    column_mapping = {

        'throughpup': 'Throughput (Mbps)',
        'throughput': 'Throughput (Mbps)',

        'delay': 'Delay (ms)',
        'delay(ms)': 'Delay (ms)',

        'jitter': 'Jitter (ms)',
        'jitter(ms)': 'Jitter (ms)',

        'sinr': 'SINR (dB)',
        'sinr(db)': 'SINR (dB)'
    }

    df = df.rename(
        columns=lambda x:
        column_mapping.get(
            x.lower(),
            x
        )
    )

    # =====================================================
    # DROP UNUSED COLUMN
    # =====================================================
    if 'Unnamed: 0' in df.columns:
        df = df.drop(
            'Unnamed: 0',
            axis=1
        )

    # =====================================================
    # AMBIL FEATURE
    # =====================================================
    data = df[
        [
            'Throughput (Mbps)',
            'Delay (ms)',
            'Jitter (ms)',
            'SINR (dB)'
        ]
    ].copy()

    print("DATA SHAPE:", data.shape)

    # =====================================================
    # HANDLE NaN & INF
    # =====================================================
    # Time series TIDAK BOLEH drop row
    # Jadi gunakan interpolasi
    # =====================================================

    for col in data.columns:

        data[col] = pd.to_numeric(
            data[col],
            errors='coerce'
        )

    # Replace inf menjadi NaN
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("\nTOTAL NaN AWAL:")
    print(data.isna().sum())

    # =====================================================
    # INTERPOLASI TIME SERIES
    # =====================================================
    data = data.interpolate(
        method='linear',
        limit_direction='both'
    )

    # Backup jika masih ada NaN
    data = data.bfill()
    data = data.ffill()

    print("\nTOTAL NaN SETELAH CLEANING:")
    print(data.isna().sum())

    print("\nDATA SHAPE SETELAH CLEANING:")
    print(data.shape)

    # =====================================================
    # TIPHON SCORING
    # =====================================================
    def score_throughput(x):

        if pd.isna(x):
            return np.nan

        if x >= 75:
            return 4

        elif x >= 50:
            return 3

        elif x >= 25:
            return 2

        else:
            return 1

    def score_delay(x):

        if pd.isna(x):
            return np.nan

        if x < 150:
            return 4

        elif x < 300:
            return 3

        elif x < 450:
            return 2

        else:
            return 1

    def score_jitter(x):

        if pd.isna(x):
            return np.nan

        if x == 0:
            return 4

        elif x < 75:
            return 3

        elif x <= 125:
            return 2

        else:
            return 1

    def score_sinr(x):

        if pd.isna(x):
            return np.nan

        if x > 20:
            return 4

        elif x >= 15:
            return 3

        elif x >= 0:
            return 2

        else:
            return 1

    # =====================================================
    # KONVERSI INDEX -> PERSENTASE
    # =====================================================
    def qos_percentage_from_index(avg_index):

        if avg_index >= 3.8:

            return 95 + (
                (avg_index - 3.8) / (4.0 - 3.8)
            ) * 5

        elif avg_index >= 3.0:

            return 75 + (
                (avg_index - 3.0) / (3.79 - 3.0)
            ) * (94.75 - 75)

        elif avg_index >= 2.0:

            return 50 + (
                (avg_index - 2.0) / (2.99 - 2.0)
            ) * (74.75 - 50)

        else:

            return 25 + (
                (avg_index - 1.0) / (1.99 - 1.0)
            ) * (49.75 - 25)

    # =====================================================
    # HITUNG QoS INDEX
    # =====================================================
    qos_raw = []

    for i in range(len(data)):

        s_t = score_throughput(data.iloc[i, 0])
        s_d = score_delay(data.iloc[i, 1])
        s_j = score_jitter(data.iloc[i, 2])
        s_s = score_sinr(data.iloc[i, 3])

        scores = [
            s for s in [s_t, s_d, s_j, s_s]
            if not np.isnan(s)
        ]

        if len(scores) == 0:

            qos_raw.append(np.nan)

        else:

            avg_index = np.mean(scores)

            qos_raw.append(
                qos_percentage_from_index(avg_index)
            )

    qos_index = np.array(qos_raw)

    qos_index = np.clip(
        qos_index,
        25,
        100
    )

    # =====================================================
    # FEATURE
    # =====================================================
    feature_values = data.values.astype(float)

    # =====================================================
    # NORMALISASI
    # =====================================================
    scaler_feat = StandardScaler()

    feature_scaled = scaler_feat.fit_transform(
        feature_values
    )

    scaler_qos = StandardScaler()

    qos_scaled = scaler_qos.fit_transform(
        qos_index.reshape(-1, 1)
    ).flatten()

    print("\nNORMALISASI SELESAI")

    # =====================================================
    # SLIDING WINDOW
    # =====================================================
    lookback = 110

    def create_dataset(
        features,
        target,
        lookback=110
    ):

        X_out = []
        y_out = []

        for i in range(
            len(features) - lookback
        ):

            window = features[
                i:i + lookback
            ]

            t_val = target[
                i + lookback
            ]

            # Tidak drop data
            # hanya skip jika masih ada NaN
            if (
                np.any(np.isnan(window))
                or np.isnan(t_val)
            ):
                continue

            X_out.append(window)
            y_out.append(t_val)

        return (
            np.array(X_out),
            np.array(y_out)
        )

    X, y = create_dataset(
        feature_scaled,
        qos_scaled,
        lookback
    )

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    # =====================================================
    # SPLIT TRAIN & VALIDATION
    # =====================================================
    split = int(0.8 * len(X))

    X_train = X[:split]
    y_train = y[:split]

    X_val = X[split:]
    y_val = y[split:]

    print("\nTRAIN SHAPE:", X_train.shape)
    print("VALIDATION SHAPE:", X_val.shape)

    # =====================================================
    # MODEL LSTM
    # =====================================================
    model = Sequential([

        Input(
            shape=(
                lookback,
                X.shape[2]
            )
        ),

        LSTM(
            128,
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(64),

        Dropout(0.3),

        Dense(1)
    ])

    # =====================================================
    # COMPILE
    # =====================================================
    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss='mse'
    )

    model.summary()

    # =====================================================
    # EARLY STOPPING
    # =====================================================
    early_stop = EarlyStopping(

        monitor='val_loss',

        patience=10,

        restore_best_weights=True,

        verbose=1
    )

    # =====================================================
    # TRAINING
    # =====================================================
    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=100,

        batch_size=32,

        callbacks=[early_stop],

        verbose=1
    )

    # =====================================================
    # PREDIKSI VALIDASI
    # =====================================================
    pred_scaled = model.predict(
        X_val
    ).flatten()

    pred_qos = scaler_qos.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()

    true_qos = scaler_qos.inverse_transform(
        y_val.reshape(-1, 1)
    ).flatten()

    pred_qos = np.clip(
        pred_qos,
        0,
        100
    )

    # =====================================================
    # EVALUASI
    # =====================================================
    rmse = np.sqrt(
        np.mean(
            (true_qos - pred_qos) ** 2
        )
    )

    mae = np.mean(
        np.abs(true_qos - pred_qos)
    )

    print("\n=== HASIL VALIDATION LSTM ===")

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    # =====================================================
    # SAVE MODEL
    # =====================================================
    model.save(
        models_dir / "model_qos_LSTM.keras"
    )

    # =====================================================
    # SAVE SCALER
    # =====================================================
    joblib.dump(
        scaler_feat,
        models_dir / "scaler_feat.pkl"
    )

    joblib.dump(
        scaler_qos,
        models_dir / "scaler_qos.pkl"
    )

    # =====================================================
    # SAVE CONFIG
    # =====================================================
    config = {

        "lookback": lookback,

        "model_type": "LSTM"
    }

    with open(
        models_dir / "config.json",
        "w"
    ) as f:

        json.dump(config, f)

    print("\nMODEL BERHASIL DISIMPAN DI FOLDER models1")

    return {

        "status": "success",

        "rmse": float(rmse),

        "mae": float(mae)
    }

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    result = train_model()

    print(result)