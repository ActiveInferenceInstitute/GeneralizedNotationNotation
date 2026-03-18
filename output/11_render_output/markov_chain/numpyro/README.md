# NUMPYRO Rendering Results

Generated from GNN POMDP Model: **Simple Markov Chain**

## Model Information

- **Model Name**: Simple Markov Chain
- **Model Description**: This model describes a minimal discrete-time Markov Chain:
- 3 states representing weather (sunny, cloudy, rainy).
- No actions — the system evolves passively.
- Observations = states directly (identity mapping for monitoring).
- Stationary transition matrix with realistic weather dynamics.
- Tests the simplest model structure: passive state evolution with no control.
- **Generation Date**: 2026-03-18 09:21:23

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 1

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 3×3 - Maps hidden states to observations
- **B Matrix (Transition)**: Present - State transitions given actions
- **C Vector (Preferences)**: Length 3 - Preferences over observations
- **D Vector (Prior)**: Length 3 - Prior beliefs over states


## Generated Files

- `Simple Markov Chain_numpyro.py` - numpyro simulation script


## Usage

Refer to the main numpyro documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: numpyro
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
