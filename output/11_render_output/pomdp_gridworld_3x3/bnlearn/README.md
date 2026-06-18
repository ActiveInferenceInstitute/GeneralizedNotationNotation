# BNLEARN Rendering Results

Generated from GNN POMDP Model: **POMDP GridWorld 3x3**

## Model Information

- **Model Name**: POMDP GridWorld 3x3
- **Model Description**: Discrete 3x3 GridWorld POMDP for strict cross-framework validation. The model has one hidden state factor with 9 grid cells, one observation modality with noisy cell observations, and one control factor with 5 boundary-clamped actions: up, down, left, right, and stay.
- **Generation Date**: 2026-06-18 07:53:05

## POMDP Dimensions

- **Number of States**: 9
- **Number of Observations**: 9
- **Number of Actions**: 5

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 9×9 - Maps hidden states to observations
- **B Matrix (Transition)**: 9×9×5 - State transitions given actions
- **C Vector (Preferences)**: Length 9 - Preferences over observations
- **D Vector (Prior)**: Length 9 - Prior beliefs over states
- **E Vector (Habits)**: Length 5 - Policy priors


## Generated Files

- `POMDP GridWorld 3x3_bnlearn.py` - bnlearn simulation script


## Usage

Refer to the main bnlearn documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: bnlearn
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
