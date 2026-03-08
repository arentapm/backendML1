import numpy as np
from scipy.linalg import svd

def mssa_decompose(series, L=50):
    N, n_features = series.shape
    L = min(L, N - 1)
    K = N - L + 1

    X = np.zeros((L * n_features, K))
    for i in range(K):
        X[:, i] = series[i:i+L].T.flatten('F')

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