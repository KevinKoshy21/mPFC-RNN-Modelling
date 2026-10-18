# Vanilla Recurrent Neural Networks

This repository contains a computational study comparing Vanilla recurrent neural networks with recurrent networks trained using an activity-correlation constraint.

## Overview

This repository contains a computational study comparing a standard recurrent neural network with a constrained recurrent neural network.

Both networks learn to reproduce multiple two-dimensional behavioral trajectories. The constrained model additionally uses a correlation-based penalty to reduce similarity between hidden activity patterns associated with different trajectories.

The notebook trains both models across multiple random seeds and evaluates:

- Behavioral performance.
- Training and validation loss.
- Predicted trajectories.
- Hidden-unit activity.
- Representational similarity matrices.
- The number of epochs required to reach different validation-loss thresholds.

## Architecture & Training
<img width="2000" height="1300" alt="model_overview_1" src="https://github.com/user-attachments/assets/f55fca06-abfe-4792-af25-ffc8e934cb67" />

The model is a recurrent network of rate units, simulated in discrete time steps. It runs in a closed loop: its own output at each step is fed back in as part of the next input. The settings for every run are in the configuration cell at the top of the notebook and saved with the results. The paper's Methods section explains the model in full.

| Component | Specification |
|---|---|
| Architecture | Discrete-time recurrent rate network with N = 200 ReLU units <!-- VERIFY: your notebook is PyTorch --> |
| Connectivity | Fixed sparse recurrent mask (p = 0.4), giving 16,000 active recurrent connections, sampled once per seed |
| Input | Static cue c<sub>k</sub> concatenated with the network's previous (x̂, ŷ) outputs (d<sub>in</sub> = 2T − 1 = 263) <!-- VERIFY --> |
| Readout | Bias-free linear map producing (x̂, ŷ) |
| Training | Closed loop; Adam, learning rate 2 × 10<sup>−4</sup>, batch size 2,000, gradient clipping at global norm 1.0 <!-- VERIFY --> |
| Regularization | λ = (0.05, 0.005, 0.05) |
| Stopping criterion | MSE threshold θ = 0.06 <!-- VERIFY: your notes say 0.02 with patience 5 --> |
| Trainable parameters | 93,200 <!-- VERIFY --> |
| Task | K = 5 trajectories, 20° spacing |
| Seeds | <!-- VERIFY: 10 or 20 matched control/constrained pairs --> |
 
The training objective is
 
<p align="center">
L = L<sub>MSE</sub> + λ<sub>1</sub> (‖W<sub>in</sub>‖ + ‖W<sub>out</sub>‖) + λ<sub>2</sub> ‖a‖ + λ<sub>3</sub> ‖W<sub>rec</sub>‖ + α P<sub>corr</sub>
</p>
where *a* is the hidden activity and P<sub>corr</sub> is the correlation penalty across task conditions. It is computed per unit: for each pair of conditions, take the correlation between the unit's tuning curves, average |r| over units, and sum over all condition pairs. Control networks use α = 0.

## Repository Structure
 <!-- TODO: check the one-line descriptions against the code in src/ -->
 
```
mPFC-RNN-Modelling/
├── notebooks/
│   └── 1_train_and_evaluate.ipynb   # Train control and constrained networks, evaluate, plot
├── src/
│   ├── config.py                    # Hyperparameters and settings
│   ├── data.py                      # Trajectory task
│   ├── model.py                     # Network and correlation penalty
│   ├── training.py                  # Training loop and stopping rule
│   ├── loading.py                   # Loading saved models and results
│   ├── analysis.py                  # Statistics and representational analyses
│   └── plotting.py                  # Figure functions
├── results/
│   ├── data/                        # Saved metrics
│   ├── figures/                     # Figure panels
│   └── models/                      # Trained weights
├── docs/
│   └── model_overview.png
├── requirements.txt
└── README.md


