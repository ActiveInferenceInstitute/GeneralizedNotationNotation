# ACTIVEINFERENCE_JL Rendering Results

Generated from GNN POMDP Model: **Bnlearn Causal Model**

## Model Information

- **Model Name**: Bnlearn Causal Model
- **Model Description**: A Bayesian Network model mapping Active Inference structure:
- S: Hidden State
- A: Action
- S_prev: Previous State
- O: Observation
- **Generation Date**: 2026-04-12 17:23:34

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

- `Bnlearn Causal Model_activeinference.jl` - activeinference_jl simulation script


## Usage

Refer to the main activeinference_jl documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: activeinference_jl
- **File Extension**: .jl
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
