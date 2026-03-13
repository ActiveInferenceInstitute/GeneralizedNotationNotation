# DISCOPY Rendering Results

Generated from GNN POMDP Model: **Hidden Markov Model Baseline**

## Model Information

- **Model Name**: Hidden Markov Model Baseline
- **Model Description**: A standard discrete Hidden Markov Model with:
- 4 hidden states with Markovian dynamics
- 6 observation symbols
- Fixed transition and emission matrices
- No action selection (passive inference only)
- Suitable for sequence modeling and state estimation tasks
- **Generation Date**: 2026-03-13 14:15:10

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

- `Hidden Markov Model Baseline_discopy.py` - discopy simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main discopy documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: discopy
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
