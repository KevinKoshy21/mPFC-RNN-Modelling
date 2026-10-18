# Vanilla Recurrent Neural Networks

This repository contains a computational study comparing Vanilla recurrent neural networks with recurrent networks trained using an activity-correlation constraint.

## Overview

## Architecture and Training

The models are recurrent neural networks with a three-channel input representing fixation, cosine, and sine directional cues. Each network contains a 200-unit recurrent hidden layer and produces two-dimensional trajectory outputs.

The recurrent dynamics use leaky integration with an integration step of `0.2`, recurrent state noise with amplitude `0.2`, and a recurrent connectivity density of `0.8`. Noise remains enabled during evaluation to match the model's intended stochastic dynamics.

The Control model is trained using trajectory-reproduction loss only. The Constrained model adds a correlation penalty on direction-specific hidden activity, weighted by `alpha = 1.0`. Both models are trained with Adam using a learning rate of `0.001`, a batch size of `2000`, and gradient clipping at `0.2`.

The main experiment uses five trajectories separated by 20 degrees and ten matched random seeds. Training stops when the validation loss reaches `0.06` for five consecutive epochs, with a maximum of `2500` epochs for the five-trajectory condition. The data use a 10% validation split and a 20% test split.


The notebook trains both models across multiple random seeds and evaluates:

- Behavioral performance.
- Training and validation loss.
- Predicted trajectories.
- Hidden-unit activity.
- Representational similarity matrices.
- The number of epochs required to reach different validation-loss thresholds.

## Architecture and Training

The model is a recurrent neural network with a three-channel input, a 300-unit recurrent hidden layer, and a two-dimensional output layer for predicting the target trajectories. Hidden states use a leaky tanh activation, recurrent Gaussian noise, and a fixed recurrent connectivity mask.

The Control model is trained using behavioral mean-squared error, while the Constrained model additionally receives a correlation penalty that encourages direction-specific hidden representations to become less similar. Both models are trained with Adam using a learning rate of `0.001`, a batch size of `64`, and gradient clipping at `0.2`.

Training uses multiple random seeds and stops when the validation MSE remains below `0.02` for five consecutive epochs. The notebook records training, validation, and test losses, model checkpoints, hidden activity, predicted trajectories.


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


