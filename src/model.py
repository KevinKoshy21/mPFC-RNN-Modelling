import numpy as np
import torch
import torch.nn as nn

from config import *


##### Vanilla (leaky, rate) RNN with a fixed sparse recurrent mask
class RNN(nn.Module):
    """
        x_{t+1} = x_t + alpha * ( -x_t + (W_hh * M) r_t + W_ih u_t + b + noise )
        r_t     = tanh(x_t)
        y_t     = W_out r_t + b_out

    The state noise is on in training mode, and in eval mode too when eval_noise=True.
    The initial state x_0 ~ U(-0.1, 0.1) is drawn in eval mode as well, so pin the
    torch RNG (EVAL_SEED) before an evaluation pass.
    """
    def __init__(self, n_in, n_h, n_out, seed=None,
                 alpha=RNN_ALPHA, noise=RNN_NOISE, density=REC_DENSITY, eval_noise=EVAL_NOISE):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.alpha = alpha
        self.noise = noise
        self.eval_noise = eval_noise
        self.W_ih  = nn.Parameter(0.1 * torch.randn(n_h, n_in))
        self.b_ih  = nn.Parameter(torch.zeros(n_h))
        self.W_hh  = nn.Parameter(0.1 * torch.randn(n_h, n_h))
        self.b_hh  = nn.Parameter(torch.zeros(n_h))
        self.W_out = nn.Parameter(0.1 * torch.randn(n_out, n_h))
        self.b_out = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("mask", (torch.rand(n_h, n_h) < density).float())
        self.n_h = n_h

    def forward(self, X):                      # X: (T, B, n_in)
        T, B, _ = X.shape
        x = (torch.rand(B, self.n_h, device=X.device) - 0.5) * 0.2
        r = torch.tanh(x)
        H, O = [], []
        for t in range(T):
            noise = self.noise * torch.randn_like(x) if (self.training or self.eval_noise) else 0.0
            dx = -x + r @ (self.W_hh * self.mask).T + X[t] @ self.W_ih.T + self.b_hh + self.b_ih + noise
            x  = x + self.alpha * dx
            r  = torch.tanh(x)
            H.append(r); O.append(r @ self.W_out.T + self.b_out)
        return torch.stack(O), torch.stack(H)   # (T, B, 2), (T, B, n_h)


##### Behavioural loss
def mse_loss(out, tgt, t0):
    """Sum of squared error from t0 onward, / (2 * B * (T - t0))."""
    return (tgt[t0:] - out[t0:]).pow(2).sum() / (2 * out.shape[1] * (out.shape[0] - t0))


def to_device(X, Y, dev):
    return X.permute(1, 0, 2).to(dev), Y.permute(1, 0, 2).to(dev)


##### Correlation penalty (unit-wise, as in the memory-based RNN's tf_corr_penalty)
def corr_penalty(hid, dirs, reduce=None, eps=1e-8):
    """Differentiable sum (or mean) over direction pairs of the unit-wise |Pearson r|.

    hid : (T, B, n_h), dirs : (B,) direction ids.
    Per direction k take the trial-mean response (n_h, T); centre each unit's time
    course and scale it by its max |.|; then for every unit correlate its time
    courses in directions k and j, take |r| and average over units.
      reduce="sum"  -> sum over the C(K,2) pairs (memory-based RNN). Default.
      reduce="mean" -> per-pair average, comparable across K.
    Directions absent from the batch are skipped; in tf_corr_penalty their pairs are 0.
    """
    reduce = PENALTY_REDUCE if reduce is None else reduce
    ks = torch.unique(dirs)
    if ks.numel() < 2:
        return hid.sum() * 0.0
    means = torch.stack([hid[:, dirs == k, :].mean(1).T for k in ks])         # (K, n_h, T)
    means = means - means.mean(dim=2, keepdim=True)
    means = means / (means.abs().amax(dim=2, keepdim=True) + eps)
    m_c   = means - means.mean(dim=2, keepdim=True)
    norms = torch.sqrt((m_c * m_c).sum(dim=2) + eps)                           # (K, n_h)
    dots  = torch.einsum("kut,jut->kju", m_c, m_c)                             # (K, K, n_h)
    corr  = dots / (norms[:, None, :] * norms[None, :, :] + eps)
    abs_corr_unit_mean = corr.abs().mean(dim=2)                                # (K, K)
    iu   = torch.triu_indices(len(ks), len(ks), offset=1)
    vals = abs_corr_unit_mean[iu[0], iu[1]]
    return vals.sum() if reduce == "sum" else vals.mean()


def corr_penalty_np(hid, dirs, reduce=None, eps=1e-8):
    """Numpy twin. MEASUREMENT only -- never differentiated."""
    reduce = PENALTY_REDUCE if reduce is None else reduce
    ks = np.unique(dirs)
    if len(ks) < 2: return 0.0
    means = np.stack([hid[:, dirs == k, :].mean(1).T for k in ks]).astype(np.float64)
    means = means - means.mean(axis=2, keepdims=True)
    means = means / (np.abs(means).max(axis=2, keepdims=True) + eps)
    m_c   = means - means.mean(axis=2, keepdims=True)
    norms = np.sqrt((m_c * m_c).sum(axis=2) + eps)
    corr  = np.einsum("kut,jut->kju", m_c, m_c) / (norms[:, None, :] * norms[None, :, :] + eps)
    v = np.abs(corr).mean(axis=2)[np.triu_indices(len(ks), 1)]
    return float(v.sum() if reduce == "sum" else v.mean())


##### Eval-mode pass over a loader
@torch.no_grad()
def collect(model, loader, dev, t0, K, want_hidden=False):
    """Eval-mode pass. Returns loss, and optionally direction-averaged hidden
    activity (K, T, n_h) plus one representative trial per direction."""
    model.eval()
    loss, nb = 0.0, 0
    sums = None; counts = None; first_trial = None
    for X, Y, Dd in loader:
        X, Y = to_device(X, Y, dev)
        o, h = model(X)
        loss += mse_loss(o, Y, t0).item(); nb += 1
        if want_hidden:
            hc = h.cpu().numpy()                     # (T, b, n_h)
            dd = Dd.numpy()
            if sums is None:
                T_, _, NH_ = hc.shape
                sums   = np.zeros((K, T_, NH_), dtype=np.float64)
                counts = np.zeros(K, dtype=np.int64)
                first_trial = np.full((K, T_, NH_), np.nan, dtype=np.float32)
            for k in range(K):
                m = dd == k
                if m.any():
                    sums[k] += hc[:, m, :].sum(1); counts[k] += int(m.sum())
                    if np.isnan(first_trial[k, 0, 0]):
                        first_trial[k] = hc[:, np.where(m)[0][0], :]
    out = {"loss": loss / max(nb, 1)}
    if want_hidden:
        assert (counts > 0).all(), f"some direction absent from this loader: {counts}"
        out["hid_dir_mean"] = (sums / counts[:, None, None]).astype(np.float32)
        out["hid_trial"]    = first_trial
        out["n_per_dir"]    = counts
    return out
