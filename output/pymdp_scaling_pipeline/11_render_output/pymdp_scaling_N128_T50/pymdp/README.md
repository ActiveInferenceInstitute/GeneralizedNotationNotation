# PYMDP Rendering Results

Generated from GNN POMDP Model: **PyMDP Scaling N128 T50**

## Model Information

- **Model Name**: PyMDP Scaling N128 T50
- **Model Description**: PyMDP runtime scaling sweep with noisy observation and stochastic transitions.
- **Generation Date**: 2026-05-06 07:04:46

## POMDP Dimensions

- **Number of States**: 128
- **Number of Observations**: 128
- **Number of Actions**: 128

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 128×128 - Maps hidden states to observations
- **B Matrix (Transition)**: 128×128×128 - State transitions given actions
- **C Vector (Preferences)**: Length 128 - Preferences over observations
- **D Vector (Prior)**: Length 128 - Prior beliefs over states


## Generated Files

- `PyMDP Scaling N128 T50_pymdp.py` - pymdp simulation script


## Usage

Refer to the main pymdp documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pymdp
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
