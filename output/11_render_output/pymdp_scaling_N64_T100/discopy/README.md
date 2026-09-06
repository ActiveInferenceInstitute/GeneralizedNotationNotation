# DISCOPY Rendering Results

Generated from GNN POMDP Model: **PyMDP Scaling N64 T100**

## Model Information

- **Model Name**: PyMDP Scaling N64 T100
- **Model Description**: PyMDP runtime scaling sweep with noisy observation and stochastic transitions.
- **Generation Date**: 2026-09-05 20:25:25

## POMDP Dimensions

- **Number of States**: 64
- **Number of Observations**: 64
- **Number of Actions**: 64

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 64×64 - Maps hidden states to observations
- **B Matrix (Transition)**: 64×64×64 - State transitions given actions
- **C Vector (Preferences)**: Length 64 - Preferences over observations
- **D Vector (Prior)**: Length 64 - Prior beliefs over states


## Generated Files

- `PyMDP_Scaling_N64_T100_discopy.py` - discopy simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main discopy documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: discopy
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
