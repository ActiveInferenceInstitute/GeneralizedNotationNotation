# ACTIVEINFERENCE_JL Rendering Results

Generated from GNN POMDP Model: **PyMDP Scaling N4 T100**

## Model Information

- **Model Name**: PyMDP Scaling N4 T100
- **Model Description**: PyMDP runtime scaling sweep with noisy observation and stochastic transitions.
- **Generation Date**: 2026-09-08 06:58:12

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

- `PyMDP_Scaling_N4_T100_activeinference.jl` - activeinference_jl simulation script


## Usage

Refer to the main activeinference_jl documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: activeinference_jl
- **File Extension**: .jl
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
