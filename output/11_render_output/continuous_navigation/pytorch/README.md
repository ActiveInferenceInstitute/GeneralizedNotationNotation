# PYTORCH Rendering Results

Generated from GNN POMDP Model: **Continuous State Navigation Agent**

## Model Information

- **Model Name**: Continuous State Navigation Agent
- **Model Description**: A continuous-state Active Inference navigation agent rendered as a native
linear-Gaussian state-space model (LGSSM):
- Hidden state x = (x, y): the continuous 2D position of the navigator.
- Observation y: noisy readings of the 2D position (identity readout).
- Control input u: a goal-seeking command added to the state each step.
- The controller closes the loop on beliefs — it pushes the filtered posterior
mean toward the preferred position goal_mean = (2.0, 2.0) with proportional
gain control_gain = 0.3, i.e. u_t = control_gain * (goal_mean - mu_t).
- **Generation Date**: 2026-09-08 06:58:15

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 2
- **Number of Actions**: 1

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `Continuous_State_Navigation_Agent_pytorch.py` - pytorch simulation script


## Usage

Refer to the main pytorch documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pytorch
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
