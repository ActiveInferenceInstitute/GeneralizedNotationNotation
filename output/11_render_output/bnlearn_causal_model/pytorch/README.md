# PYTORCH Rendering Results

Generated from GNN POMDP Model: **Bnlearn Causal Model**

## Model Information

- **Model Name**: Bnlearn Causal Model
- **Model Description**: A Bayesian Network model mapping Active Inference structure:
- S: Hidden State
- A: Action
- S_prev: Previous State
- O: Observation
- **Generation Date**: 2026-04-14 10:58:57

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 2
- **Number of Actions**: 2

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 2×2 - Maps hidden states to observations
- **B Matrix (Transition)**: 2×2×2 - State transitions given actions
- **C Vector (Preferences)**: Length 2 - Preferences over observations
- **D Vector (Prior)**: Length 2 - Prior beliefs over states


## Generated Files

- `Bnlearn Causal Model_pytorch.py` - pytorch simulation script


## Usage

Refer to the main pytorch documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pytorch
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
