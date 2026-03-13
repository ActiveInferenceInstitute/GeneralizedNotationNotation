# PYMDP Rendering Results

Generated from GNN POMDP Model: **Deep Planning Horizon POMDP**

## Model Information

- **Model Name**: Deep Planning Horizon POMDP
- **Model Description**: An Active Inference POMDP with deep (T=5) planning horizon:
- Evaluates policies over 5 future timesteps before acting
- Uses rollout Expected Free Energy accumulation
- 4 hidden states, 4 observations, 4 actions
- Each action policy is a sequence of T actions: π = [a_1, a_2, ..., a_T]
- Enables sophisticated multi-step reasoning and delayed reward attribution
- **Generation Date**: 2026-03-13 14:15:10

## POMDP Dimensions

- **Number of States**: 4
- **Number of Observations**: 4
- **Number of Actions**: 4

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 4×4 - Maps hidden states to observations
- **B Matrix (Transition)**: 4×4×4 - State transitions given actions
- **C Vector (Preferences)**: Length 4 - Preferences over observations
- **D Vector (Prior)**: Length 4 - Prior beliefs over states


## Generated Files

- `Deep Planning Horizon POMDP_pymdp.py` - pymdp simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main pymdp documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pymdp
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
