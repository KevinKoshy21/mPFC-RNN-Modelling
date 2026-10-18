import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import ttest_rel

from config import *


##################################
## Condition-averaged heatmaps  ##
##################################
def plot_sorted_averaged_activations(ax, I, activations, vmin=0.1, vmax=0.9, sorting_ref=0):
    """
    Average the activity within each condition, centre each unit and scale it by
    its max |.|, and sort the units by their peak time in condition `sorting_ref`.

    I           : (N,) condition labels
    activations : (N, n_units, T)
    returns     : (labels, (K, n_units, T) sorted, normalised averages)
    ax=None skips the drawing.
    """
    epsilon = 1e-8
    I = I.reshape(-1)
    conditions = np.unique(I)

    ##### sort order, from the reference condition
    idx = I == sorting_ref
    x = activations[idx].mean(axis=0)
    x_centered = x - x.mean(axis=1, keepdims=True)
    x_norm = x_centered / (np.abs(x_centered).max(axis=1, keepdims=True) + epsilon)
    order = np.argsort(np.argmax(x_norm, axis=1))[::-1]

    X, Labels = [], []
    for c in conditions:
        idx = I == c
        x = activations[idx].mean(axis=0)
        x_centered = x - x.mean(axis=1, keepdims=True)
        x_norm = x_centered / (np.abs(x_centered).max(axis=1, keepdims=True) + epsilon)
        X.append(x_norm[order])
        Labels.append(int(c))

    if ax is None:
        return Labels, np.array(X)
    axes = np.atleast_1d(ax)
    fig = axes[0].figure

    fig.subplots_adjust(wspace=0.3)
    for i, (x, L, ax) in enumerate(zip(X, Labels, axes)):
        im = ax.imshow(x, aspect="auto", cmap="hot",
                       extent=[0, x.shape[1], 0, x.shape[0]], vmin=vmin, vmax=vmax)
        ax.set_title(f"Cond {L+1}")
        ax.set_xlabel("Time steps")
        if i == 0:
            ax.set_ylabel("Units")
        else:
            ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.99, pad=0.02)
    cbar.set_label("Normalized Activity")
    return Labels, np.array(X)


def normalise_units(A, mode="max"):
    """
    Scale each unit's time course. A is (N_units, T).
    "max" divide by the unit's peak |.|, "z" z-score the unit, None leave raw.
    """
    A = np.asarray(A, float)
    if mode == "max":
        return A / (np.abs(A).max(axis=1, keepdims=True) + 1e-12)
    if mode == "z":
        return (A - A.mean(axis=1, keepdims=True)) / (A.std(axis=1, keepdims=True) + 1e-12)
    return A


def prepare_activations(ACTIVATIONS, normalise="max", clip=(1, 99), sort_order=None):
    """
    Normalise, sort and compute shared colour limits, once for all panels.
    Returns (A, sort_order, clim) where A is (K, N_units, T), already sorted.
    """
    A = np.array(ACTIVATIONS)
    A = np.stack([normalise_units(a, normalise) for a in A])
    # units ranked by when they peak in condition 0
    if sort_order is None:
        sort_order = np.argsort(np.argmax(A[0], axis=1))
    A = A[:, sort_order]
    clim = tuple(np.percentile(A, clip))
    return A, sort_order, clim


def plot_activation_heatmap(ax, ACTIVATIONS, i=0, sort_order=None, clim=None,
                            normalise="max", clip=(1, 99), cmap=HEATMAP_CMAP,
                            title=None, show_peaks=False, colorbar=False,
                            xlabel="Time steps", ylabel="Units", show_ylabel=True,
                            despine=False):
    """
    One condition's unit-by-time heatmap.

    ACTIVATIONS : (K, N_units, T)
    i           : which condition to draw
    sort_order  : (N_units,) row order. None -> derived from condition 0.
    clim        : (vmin, vmax). None -> percentiles of the whole array.
    returns (sort_order, clim, im) so the next panel can reuse them.
    """
    A, sort_order, full_clim = prepare_activations(
        ACTIVATIONS, normalise=normalise, clip=clip, sort_order=sort_order)
    if clim is None:
        clim = full_clim
    vmin, vmax = clim
    n_units = A.shape[1]

    im = ax.imshow(A[i], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=1, rasterized=True)
    if show_peaks:
        peaks = np.argmax(A[i], axis=1)
        ax.scatter(peaks, np.arange(n_units), s=4, c="w",
                   edgecolors="k", linewidths=0.3, zorder=2)

    ax.set_title(title if title is not None else "Cond %d" % (i + 1))
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    if despine:
        ax.spines[['right', 'top']].set_visible(False)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return sort_order, clim, im


#####################
## 2D trajectories ##
#####################
TRACE_LW = 0.5
TRACE_ALPHA = 0.5


def condition_colours(inputs):
    """One fixed colour per unique condition label, ordered low to high."""
    labels = np.unique(np.asarray(inputs).squeeze())
    return labels, {lab: COLORS[i] for i, lab in enumerate(labels)}


def _condition_slices(inputs, outputs, n_show):
    """Yield (colour, row indices) per condition, capped at n_show each."""
    if inputs is None:
        yield "k", np.arange(min(n_show, len(outputs)))
        return
    lab = np.asarray(inputs).squeeze()
    labels, colours = condition_colours(inputs)
    for c in labels:
        yield colours[c], np.where(lab == c)[0][:n_show]


def plot_trajectories_2d(ax, outputs, inputs=None, n_show=200,
                         linewidth=TRACE_LW, alpha=TRACE_ALPHA,
                         xlabel="X", ylabel="Y", title=None,
                         use_legend=True, legend_kwargs={"loc": "upper left"}):
    """
    2D trajectory fan, coloured by condition.
    outputs : (n_samples, T, 2)
    inputs  : (n_samples,) condition labels. None -> a single colour.
    n_show  : trajectories per condition.
    """
    for k, (colour, idx) in enumerate(_condition_slices(inputs, outputs, n_show)):
        name = f"C {k + 1}" if inputs is not None else None
        for j, i in enumerate(idx):
            ax.plot(outputs[i, :, 0], outputs[i, :, 1], color=colour, linewidth=linewidth,
                    alpha=alpha, label=name if j == 0 else None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    if use_legend:
        ax.legend(frameon=False, markerscale=2, alignment="left", labelspacing=0.1,
                  handlelength=0.8, **(legend_kwargs or {}))
    return ax


#########
## RSM ##
#########
def plot_rsm(RSM, title="Representational Similarity Matrix", cmap=RSM_CMAP,
             vmin=0, vmax=1, xlabel="Condition", ylabel="Condition",
             cbar_label="Mean Pearson correlation", condition_labels=None,
             annotate=True, figsize=(10, 10), dpi=300, save_path=None, ax=None):
    """Annotated (K, K) RSM with its own colorbar. Draws into `ax` if given."""
    n = RSM.shape[0]
    if condition_labels is None:
        condition_labels = [f"C{i+1}" for i in range(n)]

    owns_figure = ax is None
    if owns_figure:
        fig = plt.figure(tight_layout=True, figsize=figsize)
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    im = ax.imshow(RSM, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(condition_labels)
    ax.set_yticklabels(condition_labels)
    ax.tick_params(width=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)

    if annotate:
        threshold = (vmin + vmax) / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{RSM[i,j]:.2f}", ha="center", va="center",
                        color="white" if RSM[i, j] < threshold else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, shrink=0.85, aspect=30)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(width=0.8, length=3)
    cbar.outline.set_linewidth(0.8)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if owns_figure:
        plt.show()


##############################
## Paired boxplots + p-bars ##
##############################
def boxplot_with_lines(ax, data, xlabel=['data', 'shuffled'], colours=['green', 'gray'],
                       ylabel="Reactivation proportion", ymin=0, ymax=1, use_legend=True):
    """data : (2, n_seeds). One line per seed joins the two boxes."""
    x1 = np.ones(len(data[0]))
    x2 = x1 + 1
    for n in range(len(data.T)):
        ax.plot([1, 2], data.T[n], color="k", linewidth=0.5)

    box = ax.boxplot(data.T, showfliers=False, widths=0.5, medianprops=dict(c="k"),
                     patch_artist=True, capwidths=0.5)
    for patch in box['boxes']:
        patch.set_facecolor("white")

    ax.scatter(x1, data[0], s=20, c=colours[0], alpha=0.5, edgecolor="k", label=xlabel[0], zorder=3)
    ax.scatter(x2, data[1], s=20, c=colours[1], alpha=0.5, edgecolor="k", label=xlabel[1], zorder=3)

    ax.set_ylim(ymin, ymax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    if use_legend:
        ax.legend(loc=[0.02, 0], frameon=False, markerscale=1, alignment="left",
                  labelspacing=0.1, handlelength=0.1)
        ax.xaxis.set_visible(False)
    else:
        ax.set_xticks([1, 2], xlabel)


def plot_siginficance_symbol_paired(ax, data, stat_res, height=0.1):
    """Stars over the pair: *** p<0.001, ** p<0.01, * p<0.05, nothing otherwise."""
    bar_position = np.max(data) + np.max(data) * height
    text_position = 1.5
    p = stat_res['p']
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        return
    ax.text(text_position, bar_position, stars, ha="center", va="bottom")
    ax.hlines(bar_position, text_position - 0.4, text_position + 0.4, "k", linewidth=0.8)


def paired_ttest(data):
    """ttest_rel(control, constrained) as {'t', 'p'}; NaN when fewer than 2 seeds."""
    if data.shape[1] < 2:
        return {"t": np.nan, "p": np.nan}
    t, p = ttest_rel(data[0], data[1])
    return {"t": float(t), "p": float(p)}


def paired_boxplot(ax, data, ylabel, labels=MODEL_LABELS_SHORT, colours=PAIR):
    """boxplot_with_lines with 20 % padding and the significance bar."""
    padding = 0.2 * (data.max() - data.min())
    if padding == 0:
        padding = 0.2 * abs(data.max()) + 1e-3
    boxplot_with_lines(ax, data, xlabel=labels, ylabel=ylabel, colours=colours,
                       ymin=data.min() - padding, ymax=data.max() + padding, use_legend=False)
    stat = paired_ttest(data)
    plot_siginficance_symbol_paired(ax, data, stat, height=0.1)
    return stat


################
## Loss curves ##
################
def curve_stack(curves):
    """NaN-pad a list of 1-D curves of different lengths to (n_runs, max_len)."""
    n = max(len(c) for c in curves)
    S = np.full((len(curves), n), np.nan)
    for i, c in enumerate(curves):
        S[i, :len(c)] = c
    return S


def plot_loss_curves(axes, histories, eval_t0):
    """
    histories : {"Control": [ckpt, ...], "Constrained": [ckpt, ...]}
    Median over seeds (solid while every seed is still training, dotted after),
    interquartile band; one panel per history key.
    """
    keys = [("train_history", f"Train loss, whole trial (t ≥ {LOSS_T0}, train mode)"),
            ("val_history",   f"Validation MSE (eval mode, t ≥ {eval_t0})"),
            ("test_history",  f"Test MSE (eval mode, t ≥ {eval_t0})")]
    for ax, (key, ttl) in zip(axes, keys):
        for colour, (name, cks) in zip(PAIR, histories.items()):
            S = curve_stack([np.asarray(ck[key], float) for ck in cks])
            x = np.arange(1, S.shape[1] + 1)
            med = np.nanmedian(S, 0); q1, q3 = np.nanpercentile(S, [25, 75], axis=0)
            full = np.sum(~np.isnan(S), 0) == S.shape[0]
            ax.fill_between(x[full], q1[full], q3[full], color=colour, alpha=.20, lw=0)
            ax.plot(x[full], med[full], color=colour, lw=1.8, label=f"{name} (n={S.shape[0]})")
            if (~full).any():
                ax.plot(x[~full], med[~full], color=colour, lw=1.2, ls=":")
        if key == "val_history":
            ax.axhline(THETA_STOP, color="k", lw=1.0, ls="--")
            ax.text(0.98, 0.04, r"$\vartheta$", transform=ax.transAxes, ha="right", fontsize=9)
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("MSE"); axes[0].legend(frameon=False, fontsize=9)


##########################
## Supplementary figure ##
##########################
def panel_letter(ax, letter, x=-0.10, y=1.06, fontsize=12):
    """Bold panel letter in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="right")


def _heatmap_row(fig, row, activations, n_units, letter=None, clim=HEATMAP_CLIM):
    """Five condition heatmaps plus a shared colorbar, in one gridspec row."""
    K = len(activations)
    order = None
    for k in range(K):
        ax = fig.add_subplot(row[0, k])
        if k == 0:
            if letter:
                panel_letter(ax, letter)
            ax.set_ylim(-0.5, n_units - 0.5)              # image extent, no blank half-row
            ax.set_yticks(np.linspace(0, n_units, 7).astype(int))
            ax.invert_yaxis()
        order, clim, im = plot_activation_heatmap(
            ax, activations, k, sort_order=order, clim=clim, title=f'Cond {k+1}',
            show_ylabel=(k == 0), colorbar=False, despine=False)
    cax = fig.add_subplot(row[0, K])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(np.round(np.linspace(clim[0], clim[1], 5), 2))
    cbar.set_label("Normalised activation")


def supp_figure(act_control, act_constrained, rsm_control, rsm_constrained,
                corr, epochs, n_units, save_path=None):
    """
    Supplementary figure, 2 rows:
      top    : A  Control heatmaps (K conditions)      B  Control RSM       C  cross-condition |r| boxplot
      bottom :    Constrained heatmaps                    Constrained RSM      epochs-to-criterion boxplot

    act_*   : (K, n_units, T) condition-averaged activity of the example seed
    rsm_*   : (K, K) unit-wise RSMs of the example seed
    corr    : (2, n_seeds) cross-condition |r|, Control / Constrained
    epochs  : (2, n_seeds) epochs to criterion, Control / Constrained
    Returns (fig, {"corr": stat, "epochs": stat}).
    """
    ratios = [0.12, 0.12, 0.12, 0.12, 0.12, 0.01, 0.1, 0.25, 0.1, 0.15]
    gs  = gridspec.GridSpec(2, 6)
    fig = plt.figure(tight_layout=True, figsize=[13, 7])
    fig.subplots_adjust(hspace=0.5)
    stats = {}

    ##### top row: Control
    row = gs[0, :].subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.15)
    _heatmap_row(fig, row, act_control, n_units, letter="A")

    ax = fig.add_subplot(row[0, 7])
    plot_rsm(rsm_control, title="Control", ax=ax, cbar_label="|r|", cmap=RSM_CMAP)
    panel_letter(ax, "B")

    ax = fig.add_subplot(row[0, 9])
    stats["corr"] = paired_boxplot(ax, corr, "Cross-condition |r|")
    panel_letter(ax, "C")

    ##### bottom row: Constrained
    row = gs[1, :].subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.15)
    _heatmap_row(fig, row, act_constrained, n_units)

    ax = fig.add_subplot(row[0, 7])
    plot_rsm(rsm_constrained, title="Constrained", ax=ax, cbar_label="|r|", cmap=RSM_CMAP)

    ax = fig.add_subplot(row[0, 9])
    stats["epochs"] = paired_boxplot(ax, epochs, "Epochs")

    if save_path is not None:
        for ext in ("svg", "png", "pdf"):
            fig.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches="tight")
    return fig, stats
