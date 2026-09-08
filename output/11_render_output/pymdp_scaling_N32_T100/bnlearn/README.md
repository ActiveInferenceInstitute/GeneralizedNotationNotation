# BNLEARN Rendering Results

Generated from GNN POMDP Model: **PyMDP Scaling N32 T100**

## Model Information

- **Model Name**: PyMDP Scaling N32 T100
- **Model Description**: PyMDP runtime scaling sweep with noisy observation and stochastic transitions.
- **Generation Date**: 2026-09-08 06:58:12

## POMDP Dimensions

- **Number of States**: 32
- **Number of Observations**: 32
- **Number of Actions**: 32

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 32×32 - Maps hidden states to observations
- **B Matrix (Transition)**: 32×32×32 - State transitions given actions
- **C Vector (Preferences)**: Length 32 - Preferences over observations
- **D Vector (Prior)**: Length 32 - Prior beliefs over states


## Generated Files

- `PyMDP_Scaling_N32_T100_bnlearn.py` - bnlearn simulation script


## Usage

Refer to the main bnlearn documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: bnlearn
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
