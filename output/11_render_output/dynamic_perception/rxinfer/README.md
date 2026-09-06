# RXINFER Rendering Results

Generated from GNN POMDP Model: **Dynamic Perception Model**

## Model Information

- **Model Name**: Dynamic Perception Model
- **Model Description**: A dynamic perception model extending the static model with temporal dynamics:
- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via recognition matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations
- **Generation Date**: 2026-09-05 20:32:38

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 2
- **Number of Actions**: 1

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 2×2 - Maps hidden states to observations
- **B Matrix (Transition)**: 2×2 - State transitions given actions
- **C Vector (Preferences)**: Length 2 - Preferences over observations
- **D Vector (Prior)**: Length 2 - Prior beliefs over states


## Generated Files

- `Dynamic_Perception_Model_rxinfer.jl` - rxinfer simulation script


## Usage

Refer to the main rxinfer documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: rxinfer
- **File Extension**: .jl
- **Multi-Modality Support**: ❌
- **Multi-Factor Support**: ❌
