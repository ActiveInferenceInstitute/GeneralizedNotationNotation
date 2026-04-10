# NUMPYRO Rendering Results

Generated from GNN POMDP Model: **Hidden Markov Model Baseline**

## Model Information

- **Model Name**: Hidden Markov Model Baseline
- **Model Description**: A standard discrete Hidden Markov Model with:
- 4 hidden states with Markovian dynamics
- 6 observation symbols
- Fixed transition and emission matrices
- No action selection (passive inference only)
- Suitable for sequence modeling and state estimation tasks
- **Generation Date**: 2026-04-10 10:25:04

## POMDP Dimensions

- **Number of States**: 4
- **Number of Observations**: 6
- **Number of Actions**: 4

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 6×4 - Maps hidden states to observations
- **B Matrix (Transition)**: Present - State transitions given actions
- **C Vector (Preferences)**: Length 6 - Preferences over observations
- **D Vector (Prior)**: Length 4 - Prior beliefs over states


## Generated Files

- `Hidden Markov Model Baseline_numpyro.py` - numpyro simulation script


## Usage

Refer to the main numpyro documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: numpyro
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
