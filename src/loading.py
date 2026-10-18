import numpy as np
import torch

from config import *
from model import RNN
from training import ckpt_path


def load_ckpt(path):
    """torch.load with a weights_only fallback for older/newer torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def mse_trialweighted(pred, targ, t0):
    """Training's normalisation (sum sq / (2 * B * (T - t0))), over all trials at once."""
    d = targ[:, t0:, :] - pred[:, t0:, :]
    return float((d ** 2).sum() / (2 * pred.shape[0] * (pred.shape[1] - t0)))


def mse_batchmean(pred, targ, t0, bs=BATCH_SIZE):
    """What training logged: mean over batches of per-batch means."""
    return float(np.mean([mse_trialweighted(pred[i:i + bs], targ[i:i + bs], t0)
                          for i in range(0, len(pred), bs)]))


@torch.no_grad()
def extract(D, idx, models_dir, seed, cond, eval_seed=EVAL_SEED, chunk=BATCH_SIZE, keep_hidden=True):
    """
    Re-load one checkpoint and run it (eval mode) on the trials `idx` of D.

    Returns predictions (N, T, 2), labels (N,), hidden (N, n_units, T) and metadata.
    The labels are direction ids 0..K-1; direction k is trajectory / condition C{k+1}.
    """
    K   = D["K"]
    p   = ckpt_path(models_dir, K, seed, cond)
    assert p.exists(), f"missing checkpoint: {p}"
    ck  = load_ckpt(p)
    assert ck["K"] == K and ck["seed"] == seed and ck["cond"] == cond
    assert ck.get("loss_t0") == LOSS_T0, f"{p.name}: trained with loss_t0={ck.get('loss_t0')}, expected {LOSS_T0}"

    m = RNN(ck["n_in"], ck["n_h"], ck["n_out"], alpha=ck["rnn_alpha"], noise=ck["rnn_noise"],
            eval_noise=ck.get("eval_noise", False)).to(DEV)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()

    stim = torch.from_numpy(D["stim"][idx])
    torch.manual_seed(eval_seed)                      # pins the random initial state
    P, H = [], []
    for i in range(0, len(idx), chunk):
        X = stim[i:i + chunk].permute(1, 0, 2).to(DEV)
        o, h = m(X)
        P.append(o.permute(1, 0, 2).cpu().numpy())    # (b, T, 2)
        if keep_hidden:
            H.append(h.permute(1, 2, 0).cpu().numpy())  # (b, n_h, T)

    pred   = np.concatenate(P, 0).astype(np.float32)
    hidden = np.concatenate(H, 0).astype(np.float32) if keep_hidden else None
    labels = D["dirs"][idx].astype(np.int64)
    targ   = D["targ"][idx]
    go     = ck.get("eval_t0", D["GO_STEP"])   # window the checkpoint was validated on

    out = dict(seed=seed, cond=cond, model=LAB[cond], pred=pred, labels=labels,
               targ=targ, hidden=hidden, ckpt=ck,
               stop_epoch=int(ck["stop_epoch"]),
               val_history=np.asarray(ck["val_history"], float),
               mse_trial=mse_trialweighted(pred, targ, go),
               mse_batch=mse_batchmean(pred, targ, go))
    if keep_hidden:
        out["hidden_dir_mean"] = np.stack([hidden[labels == k].mean(0) for k in range(K)])
    return out
