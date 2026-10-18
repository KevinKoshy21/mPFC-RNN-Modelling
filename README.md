# Vanilla Recurrent Neural Networks

This repository compares a standard recurrent neural network (Control) and with one trained with an additional activity-correlation constraint (Constrained). It tests whether the constraint changes how quickly networks learn and how similar their hidden representations are across movement directions.

## Architecture and Training

Both models are leaky tanh recurrent neural networks with 200 hidden units. They receive fixation and directional information as input and produce two-dimensional behavioral trajectories.

The Control model is trained using mean squared error (MSE). The Constrained model uses the same objective together with an additional penalty that encourages more distinct hidden representations for different directions. Both models are trained with Adam.

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


