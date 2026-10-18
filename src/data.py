import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import gaussian_filter1d

from config import *


def generate_diverging_trajectories(
    num_trajectories=8, total_angle_deg=120, d1=10, d2=5,
    n_straight=50, n_arc=50, n_hold_origin=20, n_hold_target=20,
    augmentation_factor=10, noise_level=0.02, plot=True, figsize=(18, 6),
    n_show=40, random_seed=42, scale=0.5, sigma=2, save_Fig=None):
    """
    Trajectories that hold at the origin, move along a straight ray at a
    condition-specific angle, then hold at the target.

    Returns
    -------
    inputs  : (N, 1)     condition labels, k * scale
    outputs : (N, T, 2)  smoothed (x, y) trajectories
    angles  : (K,)       ray angles in degrees
    """
    np.random.seed(random_seed)
    angles_deg = np.linspace(-total_angle_deg / 2, total_angle_deg / 2, num_trajectories)

    trajectories = []
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        x0 = np.zeros(n_hold_origin); y0 = np.zeros(n_hold_origin)
        y1 = np.linspace(0, d1, n_straight); x1 = np.zeros_like(y1)
        t  = np.linspace(0, 1, n_arc)
        x2 = d2 * t * np.sin(angle_rad)
        y2 = d1 + d2 * t * np.cos(angle_rad)
        x3 = np.ones(n_hold_target) * x2[-1]
        y3 = np.ones(n_hold_target) * y2[-1]
        trajectories.append((np.concatenate([x0, x1, x2, x3]),
                             np.concatenate([y0, y1, y2, y3])))
    X = np.array([t_[0] for t_ in trajectories])
    Y = np.array([t_[1] for t_ in trajectories])
    labels = np.arange(num_trajectories) * scale
    data_variance = noise_level * np.max(np.abs(X))
    for _ in range(augmentation_factor):
        X = np.concatenate((X, X + data_variance * np.random.uniform(-1, 1, size=X.shape)), axis=0)
        Y = np.concatenate((Y, Y + data_variance * np.random.uniform(-1, 1, size=Y.shape)), axis=0)
        labels = np.concatenate((labels, labels), axis=0)
    outputs = gaussian_filter1d(np.stack((X, Y), axis=2), sigma=sigma, axis=1)
    inputs  = labels.reshape(-1, 1)

    if plot:
        time_ax = np.arange(outputs.shape[1])
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        cols = plt.cm.autumn(np.linspace(0, 1, num_trajectories))
        lab_ids = np.rint(inputs[:, 0] / scale).astype(int)
        ax = axes[0]
        for i in range(min(n_show, outputs.shape[0])):
            ax.plot(outputs[i, :, 0], outputs[i, :, 1], alpha=0.7, color=cols[lab_ids[i]], lw=0.8)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title("2D trajectories")
        for j, nm in [(1, "X"), (2, "Y")]:
            ax = axes[j]
            for i in range(min(n_show, outputs.shape[0])):
                ax.plot(time_ax, outputs[i, :, j - 1], alpha=0.7, color=cols[lab_ids[i]], lw=0.8)
            ax.axvspan(0, n_hold_origin, color="blue", alpha=.15, label="Sampling")
            ax.axvspan(n_hold_origin, outputs.shape[1] - n_hold_target, color="gray", alpha=.15, label="Movement")
            ax.axvspan(outputs.shape[1] - n_hold_target, outputs.shape[1], color="green", alpha=.15, label="Reward")
            ax.grid(alpha=.3); ax.set_xlabel("Time"); ax.set_ylabel(nm); ax.set_title(f"{nm}(t)")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        if save_Fig: plt.savefig(save_Fig, dpi=200, bbox_inches="tight")
        plt.show()
    return inputs, outputs, angles_deg


def build_data(K, spacing_deg=SPACING_DEG, plot=False):
    """All data for one K. Returns a dict; nothing global is mutated."""
    total_angle = spacing_deg * (K - 1)
    INPUTS, OUTPUTS, angles = generate_diverging_trajectories(
        num_trajectories=K, total_angle_deg=total_angle, d1=D1, d2=D2,
        n_hold_origin=N_HOLD_ORIGIN, n_hold_target=N_HOLD_TARGET,
        n_straight=N_STRAIGHT, n_arc=N_ARC, augmentation_factor=AUG_FACTOR,
        noise_level=NOISE_LEVEL, plot=plot, figsize=(16, 4), n_show=250,
        scale=SCALE, sigma=SIGMA, random_seed=DATA_SEED)

    T_TOTAL, GO_STEP = OUTPUTS.shape[1], N_HOLD_ORIGIN
    dir_ids = np.rint(INPUTS[:, 0] / SCALE).astype(np.int64)
    assert set(np.unique(dir_ids)) == set(range(K)), "direction id recovery failed"

    templates = np.stack([OUTPUTS[dir_ids == d][:1].mean(0) for d in range(K)])
    end_xy    = templates[:, -1, :]
    theta_rad = np.arctan2(end_xy[:, 1], end_xy[:, 0])
    theta_deg = np.rad2deg(theta_rad)
    realised  = np.diff(theta_deg)   
    assert np.allclose(np.abs(realised), spacing_deg, atol=0.5), \
        f"realised spacing {np.round(realised,3)} != requested {spacing_deg}"

    N_trials = OUTPUTS.shape[0]
    if INPUT_MODE == "paper":
        stim = np.zeros((N_trials, T_TOTAL, 3), dtype=np.float32)
        stim[:, :GO_STEP, 0] = FIX_LEVEL * FIX_SCALE
        stim[:, :, 1] = (CUE_LEVEL * np.cos(theta_rad))[dir_ids][:, None]
        stim[:, :, 2] = (CUE_LEVEL * np.sin(theta_rad))[dir_ids][:, None]
        N_IN = 3
    else:
        stim = np.repeat(INPUTS[:, None, :], T_TOTAL, axis=1).astype(np.float32)
        N_IN = 1
    targ, dirs = OUTPUTS.astype(np.float32), dir_ids.astype(np.int64)

    if N_CAP is not None and N_trials > N_CAP:   
        per_dir = N_CAP // K
        rng = np.random.default_rng(0)
        keep = np.concatenate([rng.choice(np.where(dirs == d)[0], size=per_dir, replace=False)
                               for d in range(K)])
        rng.shuffle(keep)
        stim, targ, dirs = stim[keep], targ[keep], dirs[keep]

    return dict(K=K, stim=stim, targ=targ, dirs=dirs, N_IN=N_IN,
                T_TOTAL=T_TOTAL, GO_STEP=GO_STEP, theta_deg=theta_deg,
                angles=angles, outputs_full=OUTPUTS, dir_ids_full=dir_ids)


class RNNDataset(Dataset):
    def __init__(self, inputs, targets, dir_labels):
        self.inputs     = torch.from_numpy(inputs).float()
        self.targets    = torch.from_numpy(targets).float()
        self.dir_labels = torch.from_numpy(dir_labels).long()
    def __len__(self):  return self.inputs.shape[0]
    def __getitem__(self, i): return self.inputs[i], self.targets[i], self.dir_labels[i]


def make_loaders(D, val_frac=VAL_FRAC, test_frac=TEST_FRAC,
                 batch_size=BATCH_SIZE, seed=SPLIT_SEED):
    """Split is a function of SPLIT_SEED only -> identical for every model seed,
    so a seed varies network init and nothing else."""
    stim, targ, dirs = D["stim"], D["targ"], D["dirs"]
    N   = stim.shape[0]
    idx = np.random.RandomState(seed).permutation(N)
    n_test     = int(test_frac * N)
    test_idx   = idx[N - n_test:]
    tv_idx     = idx[:N - n_test]

    ds_tv   = RNNDataset(stim[tv_idx],   targ[tv_idx],   dirs[tv_idx])
    ds_test = RNNDataset(stim[test_idx], targ[test_idx], dirs[test_idx])
    n_val   = int(val_frac * len(ds_tv))
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds_tv, [len(ds_tv) - n_val, n_val], generator=g)

    return dict(
        train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        val  =DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        test =DataLoader(ds_test,  batch_size=batch_size, shuffle=False),
        test_idx=test_idx, dirs_test=dirs[test_idx])


def split_indices(D, seed=SPLIT_SEED):
    """Mirrors make_loaders(): same RandomState, same torch generator, same order."""
    N   = D["stim"].shape[0]
    idx = np.random.RandomState(seed).permutation(N)
    n_test   = int(TEST_FRAC * N)
    test_idx = idx[N - n_test:]
    tv_idx   = idx[:N - n_test]
    n_val = int(VAL_FRAC * len(tv_idx))
    g = torch.Generator().manual_seed(seed)
    tr_sub, va_sub = random_split(range(len(tv_idx)), [len(tv_idx) - n_val, n_val], generator=g)
    return dict(train=tv_idx[np.asarray(tr_sub.indices)], val=tv_idx[np.asarray(va_sub.indices)],
                test=test_idx, all=np.arange(N))
