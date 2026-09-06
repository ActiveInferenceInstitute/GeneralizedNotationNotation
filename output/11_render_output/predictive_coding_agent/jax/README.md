# JAX Rendering Results

Generated from GNN POMDP Model: **Predictive Coding Active Inference Agent**

## Model Information

- **Model Name**: Predictive Coding Active Inference Agent
- **Model Description**: A continuous predictive-coding Active Inference agent rendered as a native
linear-Gaussian state-space model (LGSSM). The agent runs passively — it has no
control input:
- Hidden state mu = (mu, mu_dot): the generalized coordinates of the belief.
- Observation y: an identity readout of both generalized coordinates.
- F encodes the linearized dynamics f(mu): mu integrates mu_dot (dt = 0.1) and
mu_dot leaks toward the flow.
- Q and R are the dynamics- and sensory-error covariances (the inverse
precisions of the predictive-coding formulation).
- **Generation Date**: 2026-09-05 20:25:31

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 2
- **Number of Actions**: 0

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `Predictive_Coding_Active_Inference_Agent_jax.py` - jax simulation script


## Usage

Refer to the main jax documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: jax
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
