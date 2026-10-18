import time
from pathlib import Path

import numpy as np
import torch

from config import *
from model import RNN, mse_loss, to_device, corr_penalty, corr_penalty_np, collect


def tag_of(K, seed, cond):
    return f"K{K}_s{seed}_{cond}"


def ckpt_path(models_dir, K, seed, cond):
    return Path(models_dir) / f"{tag_of(K, seed, cond)}.pt"


def train_run(D, L, seed, cond, models_dir, print_every=100, t_start=None):
    """
    Train one network until validation MSE stays <= THETA_STOP for STOP_PATIENCE
    consecutive epochs (or MAX_EPOCHS_BY_K[K] is hit -> censored run), then save
    a checkpoint holding the weights, the loss histories and the final metrics.

    cond = "std" -> Control      : MSE only
    cond = "con" -> Constrained  : MSE + ALPHA * corr_penalty
    """
    K = D["K"]
    t0_train = LOSS_T0          # training objective window (0 = whole trial)
    t0_eval  = EVAL_T0          # val / test / stopping window (0 = whole trial)
    max_ep  = MAX_EPOCHS_BY_K[K]
    use_pen = (cond == "con")

    m   = RNN(D["N_IN"], N_H, 2, seed=seed).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    log = {"tr": [], "val": [], "test": [], "pen": []}
    incomplete_batches = 0
    streak, stop_ep = 0, None

    for ep in range(max_ep):
        m.train(); tr = pen = 0.0; nb = 0
        for X, Y, Dd in L["train"]:
            X, Y = to_device(X, Y, DEV); Dd = Dd.to(DEV)
            if Dd.unique().numel() < K:
                incomplete_batches += 1
            opt.zero_grad()
            o, h = m(X)
            l = mse_loss(o, Y, t0_train)
            p = corr_penalty(h, Dd)
            (l + ALPHA * p if use_pen else l).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step()
            tr += l.item(); pen += p.item(); nb += 1

        vl = collect(m, L["val"], DEV, t0_eval, K)["loss"]
        te = collect(m, L["test"], DEV, t0_eval, K)["loss"]
        log["tr"].append(tr / nb); log["pen"].append(pen / nb)
        log["val"].append(vl); log["test"].append(te)

        if print_every and (ep % print_every == 0):
            el = f"  [{(time.time()-t_start)/60:.1f} min]" if t_start else ""
            print(f"    K{K} s{seed} {cond}  ep {ep+1:5d}/{max_ep}   "
                  f"MSE {tr/nb:.5f}  val {vl:.5f}  test {te:.5f}  "
                  f"|r|_{PENALTY_REDUCE} {pen/nb:7.3f}{el}", flush=True)

        if THETA_STOP is not None:
            streak = streak + 1 if vl <= THETA_STOP else 0
            if streak >= STOP_PATIENCE:
                stop_ep = ep + 2 - STOP_PATIENCE        # first epoch of the streak, 1-indexed
                print(f"    K{K} s{seed} {cond}  STOP at ep {stop_ep}/{max_ep}  "
                      f"val {vl:.6f} <= theta {THETA_STOP:.6f}", flush=True)
                break
    else:
        if THETA_STOP is not None:
            print(f"    !! K{K} s{seed} {cond}  hit the {max_ep}-epoch CAP without reaching "
                  f"theta={THETA_STOP:.6f} (best val {min(log['val']):.6f}) -- CENSORED.", flush=True)

    ##### final eval-mode metrics, t >= EVAL_T0
    tr_f = collect(m, L["train"], DEV, t0_eval, K)
    va_f = collect(m, L["val"], DEV, t0_eval, K)
    te_f = collect(m, L["test"], DEV, t0_eval, K, want_hidden=True)

    Hcorr = te_f["hid_dir_mean"]
    Ccorr = np.corrcoef(Hcorr.reshape(K, -1))
    iu = np.triu_indices(K, 1)
    mean_abs_r = float(np.mean(np.abs(Ccorr[iu]))) if K > 1 else np.nan
    mean_signed_r = float(np.mean(Ccorr[iu])) if K > 1 else np.nan

    checkpoint = {
        "model_state_dict": {name: tensor.detach().cpu() for name, tensor in m.state_dict().items()},
        "K": K, "seed": seed, "cond": cond,
        "n_in": D["N_IN"], "n_h": N_H, "n_out": 2,
        "rnn_alpha": m.alpha, "rnn_noise": m.noise, "eval_noise": m.eval_noise,
        "alpha": ALPHA, "penalty_reduce": PENALTY_REDUCE,
        "lr": LR, "clip": CLIP, "batch_size": BATCH_SIZE,
        "loss_t0": LOSS_T0, "eval_t0": EVAL_T0,
        "theta_stop": THETA_STOP, "stop_patience": STOP_PATIENCE,
        "max_epochs": max_ep,
        "stop_epoch": stop_ep if stop_ep is not None else len(log["val"]),
        "reached_theta": (stop_ep is not None) or THETA_STOP is None,
        "epochs_run": len(log["val"]),
        "incomplete_batches": incomplete_batches,
        "train_mse": float(tr_f["loss"]),
        "val_mse": float(va_f["loss"]),
        "test_mse": float(te_f["loss"]),
        "mean_abs_pearson_r": mean_abs_r,
        "mean_signed_pearson_r": mean_signed_r,
        "mean_abs_unit_r": corr_penalty_np(te_f["hid_dir_mean"].transpose(1, 0, 2),
                                           np.arange(K), reduce="mean"),     # what the penalty acts on
        "corr_penalty_sum": corr_penalty_np(te_f["hid_dir_mean"].transpose(1, 0, 2),
                                            np.arange(K), reduce="sum"),
        "train_history": np.asarray(log["tr"], dtype=np.float32),       # train mode, t >= LOSS_T0
        "val_history": np.asarray(log["val"], dtype=np.float32),        # eval mode, t >= EVAL_T0
        "test_history": np.asarray(log["test"], dtype=np.float32),
        "penalty_history": np.asarray(log["pen"], dtype=np.float32),
    }
    path = ckpt_path(models_dir, K, seed, cond)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def run_study(D, L, models_dir, seeds=SEEDS, conds=CONDS, print_every=100):
    """Train every (seed, cond) pair whose checkpoint is not on disk yet."""
    K = D["K"]
    t_start = time.time(); n_new = 0
    todo = [(s, c) for s in seeds for c in conds if not ckpt_path(models_dir, K, s, c).exists()]
    print(f"{len(seeds)*len(conds) - len(todo)} cached, {len(todo)} to train.", flush=True)

    for seed, cond in todo:
        print(f"\n=== K={K}  seed={seed}  {LAB[cond]} ===", flush=True)
        ck = train_run(D, L, seed, cond, models_dir, print_every=print_every, t_start=t_start)
        n_new += 1
        el = (time.time() - t_start) / 60
        how = (f"stopped at ep {ck['stop_epoch']}" if ck["reached_theta"]
               else f"CENSORED at cap {ck['max_epochs']}")
        print(f"  -> {how}. val {ck['val_history'][-1]:.6f}  test_eval {ck['test_mse']:.5f}  "
              f"[{el:.1f} min elapsed, ~{el/n_new*(len(todo)-n_new):.0f} min left]", flush=True)
    print(f"\nfinished in {(time.time()-t_start)/60:.1f} min", flush=True)
