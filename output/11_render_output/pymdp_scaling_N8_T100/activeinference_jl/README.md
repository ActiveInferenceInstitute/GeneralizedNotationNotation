# ACTIVEINFERENCE_JL Rendering Results

Generated from GNN POMDP Model: **PyMDP Scaling N8 T100**

## Model Information

- **Model Name**: PyMDP Scaling N8 T100
- **Model Description**: PyMDP runtime scaling sweep with noisy observation and stochastic transitions.
- **Generation Date**: 2026-09-05 20:25:21

## POMDP Dimensions

- **Number of States**: 8
- **Number of Observations**: 8
- **Number of Actions**: 8

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 8×8 - Maps hidden states to observations
- **B Matrix (Transition)**: 8×8×8 - State transitions given actions
- **C Vector (Preferences)**: Length 8 - Preferences over observations
- **D Vector (Prior)**: Length 8 - Prior beliefs over states


## Generated Files

- `PyMDP_Scaling_N8_T100_activeinference.jl` - activeinference_jl simulation script


## Usage

Refer to the main activeinference_jl documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: activeinference_jl
- **File Extension**: .jl
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
