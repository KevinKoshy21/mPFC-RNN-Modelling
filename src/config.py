import torch
import colorcet as cc
from matplotlib.colors import ListedColormap


############################
## Study Design           ##
############################
K_MAIN        = 5                                    # trajectories
SPACING_DEG   = 20                                   # degrees between adjacent trajectories
SEEDS         = [42, 123, 456, 777, 999,
                 2023, 31415, 271828, 7, 13]         # matched-init pairs (Control / Constrained share a seed)
CONDS         = ["std", "con"]                       # std = Control (MSE only), con = Constrained (MSE + penalty)
LAB           = {"std": "Control", "con": "Constrained"}

############################
## Objective              ##
############################
ALPHA          = 1.0        # penalty weight
PENALTY_REDUCE = "sum"      # "sum" over direction pairs | "mean" (pair-count normalised)

############################
## Model                  ##
############################
N_H           = 200         # hidden units
# The supplementary-figure run (hash 6231413f5c) was trained with 0.2 / 0.2; the last edit of the
# training notebook had 0.4 / 0.8 (a later, unfinished run). See CHANGES.md and REPORT.md.
RNN_ALPHA     = 0.2         # integration step  x <- x + alpha * dx
RNN_NOISE     = 0.2         # state-noise amplitude
EVAL_NOISE    = True        # keep the state noise on in eval mode (validation, test, analysis), as in the memory-based RNN
REC_DENSITY   = 0.8         # fraction of non-zero recurrent connections

############################
## Optimisation           ##
############################
LR            = 1e-3
CLIP          = 0.2
BATCH_SIZE    = 2000
MAX_EPOCHS_BY_K = {2: 1200, 3: 1500, 4: 2000, 5: 2500, 6: 3500}

############################
## Criterion-stopped runs ##
############################
THETA_STOP    = 0.06        
STOP_PATIENCE = 5           # consecutive epochs at or below THETA_STOP required to stop

############################
## Loss window            ##
############################
# Training MSE starts at LOSS_T0, validation / test / THETA_STOP at EVAL_T0

LOSS_T0       = 0
EVAL_T0       = 0

############################
## Data Generator         ##
############################
TOTAL         = 100
D1, D2        = 0, 10
N_HOLD_ORIGIN = int(TOTAL / 6)                 # 16  -- sampling epoch
N_HOLD_TARGET = int(TOTAL / 6)                 # 16  -- reward epoch
N_STRAIGHT    = int(TOTAL * (0.1 * D1))        # 0
N_ARC         = int(TOTAL * (0.1 * D2))        # 100 -- movement epoch
SCALE         = 1.0
NOISE_LEVEL   = 0.05
AUG_FACTOR    = 13
SIGMA         = 3
DATA_SEED     = 42
SPLIT_SEED    = 42          
N_CAP         = None        
VAL_FRAC, TEST_FRAC = 0.1, 0.2
INPUT_MODE    = "paper"     # 3-channel (fixation, cos, sin) input
FIX_SCALE, FIX_LEVEL, CUE_LEVEL = 0.1, 2.0, 2.0

############################
## Evaluation             ##
############################
EVAL_SEED     = 0           
SPLIT         = "test"      
SEED_FIG      = SEEDS[0]    

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################
## Colors & Labels ##
#####################
COLORS        = ["#c8712d", "#d9a46a", "#a8a8a8", "#7597b5", "#2e5f8a"]
PAIR          = [COLORS[0], COLORS[-1]]          # Control, Constrained
RSM_CMAP      = ListedColormap(COLORS)
HEATMAP_CMAP  = cc.cm.fire
HEATMAP_CLIM  = (0.3, 0.99)
MODEL_LABELS_SHORT = ("Contr", "Constr")


###########################
## Configuration Summary ##
###########################
def settings() -> dict:
    return {
        "task": {
            "K": K_MAIN,
            "spacing_deg": SPACING_DEG,
            "num_steps": N_HOLD_ORIGIN + N_STRAIGHT + N_ARC + N_HOLD_TARGET,
            "n_hold_origin": N_HOLD_ORIGIN,
            "n_movement": N_STRAIGHT + N_ARC,
            "n_hold_target": N_HOLD_TARGET,
            "augmentation_factor": AUG_FACTOR,
            "n_trials": K_MAIN * 2 ** AUG_FACTOR if N_CAP is None else N_CAP,
            "noise_level": NOISE_LEVEL,
            "sigma": SIGMA,
            "input_mode": INPUT_MODE,
        },
        "model": {
            "hidden_units": N_H,
            "rnn_alpha": RNN_ALPHA,
            "rnn_noise": RNN_NOISE,
            "eval_noise": EVAL_NOISE,
            "rec_density": REC_DENSITY,
        },
        "training": {
            "alpha_penalty": ALPHA,
            "penalty_reduce": PENALTY_REDUCE,
            "learning_rate": LR,
            "grad_clip_norm": CLIP,
            "batch_size": BATCH_SIZE,
            "loss_t0": LOSS_T0,
            "eval_t0": EVAL_T0,
            "theta_stop": THETA_STOP,
            "stop_patience": STOP_PATIENCE,
            "max_epochs": MAX_EPOCHS_BY_K[K_MAIN],
            "seeds": SEEDS,
        },
    }


def print_config(config, indent=0):
    for key, value in config.items():
        prefix = " " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 6)
        else:
            print(f"{prefix}{key}: {value}")
    print("")
