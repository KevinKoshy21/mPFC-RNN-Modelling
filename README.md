# Control vs Constrained Recurrent Neural Networks

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


