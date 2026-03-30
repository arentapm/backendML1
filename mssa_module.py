# =========================================================
# MSSA MODULE (HARUS IDENTIK DENGAN TRAINING)
# =========================================================
import numpy as np
from scipy.linalg import svd
import pandas as pd

def mssa_decompose(series, L=50):

    N, n_features = series.shape

    # ⚠️ HARUS SAMA DENGAN TRAINING
    L = min(L, N // 3)

    K = N - L + 1

    # =========================
    # EMBEDDING MATRIX
    # =========================
    X = np.zeros((L * n_features, K))

    for i in range(K):
        X[:, i] = series[i:i+L].T.flatten('F')

    # =========================
    # SVD
    # =========================
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

    # =========================
    # RECONSTRUCTION
    # =========================
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