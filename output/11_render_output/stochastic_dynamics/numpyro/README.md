# NUMPYRO Rendering Results

Generated from GNN POMDP Model: **Stochastic Continuous Dynamics Agent**

## Model Information

- **Model Name**: Stochastic Continuous Dynamics Agent
- **Model Description**: A continuous-state Active Inference agent whose dynamics carry explicit process
and observation noise, rendered as a native linear-Gaussian state-space model
(LGSSM). The agent runs passively — it has no control input:
- Hidden state x = (position, velocity): the Euler-discretized (dt = 0.1) SDE.
- Observation y: two noisy readouts, both reading the position.
- Q is the process-noise covariance (inverse process precision); R is the
observation-noise covariance (inverse observation precision).
- **Generation Date**: 2026-09-05 20:25:31

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 2
- **Number of Actions**: 0

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `Stochastic_Continuous_Dynamics_Agent_numpyro.py` - numpyro simulation script


## Usage

Refer to the main numpyro documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: numpyro
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
