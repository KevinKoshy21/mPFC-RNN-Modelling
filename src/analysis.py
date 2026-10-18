import numpy as np
from scipy.stats import pearsonr

from config import STOP_PATIENCE


def dir_mean(hidden, labels, K):
    """(N, n_units, T) -> (K, n_units, T), trial-averaged within each direction."""
    return np.stack([hidden[labels == k].mean(0) for k in range(K)])


def compute_rsm(hidden):
    """
    Unit-wise RSM: for every pair of conditions, the |Pearson r| between the two
    time courses of each unit, averaged over units.

    hidden : (conditions, neurons, time)   e.g. (5, 300, 132)
    returns: (conditions, conditions)
    """
    n_cond, n_units, _ = hidden.shape
    RSM = np.zeros((n_cond, n_cond))
    for i in range(n_cond):
        for j in range(i, n_cond):
            corr = np.empty(n_units)
            for u in range(n_units):
                corr[u] = np.abs(pearsonr(hidden[i, u], hidden[j, u])[0])
            mean_corr = np.nanmean(corr)
            RSM[i, j] = mean_corr
            RSM[j, i] = mean_corr
    return RSM


def rsm(hidden_dir_mean):
    """
    Population RSM over the K directions, as the training penalty sees it: each
    direction's trial-averaged response is flattened time-major (T x n_units) and
    all pairs are correlated. Returns the signed (K, K) Pearson matrix.
    """
    K = hidden_dir_mean.shape[0]
    M = hidden_dir_mean.transpose(0, 2, 1).reshape(K, -1)     # (K, T * n_units)
    return np.corrcoef(M)


def offdiag_mean_abs(C):
    """Mean |r| over the strict upper (== lower) triangle."""
    iu = np.triu_indices(C.shape[0], 1)
    return float(np.abs(C[iu]).mean())


def first_crossing(curve, level, patience=STOP_PATIENCE):
    """First epoch (1-indexed) that starts `patience` consecutive epochs <= level.
    Same rule as the training stop, so at THETA_STOP it must equal stop_epoch."""
    c = np.asarray(curve, float) <= level
    for i in range(len(c) - patience + 1):
        if c[i:i + patience].all():
            return i + 1
    return None
