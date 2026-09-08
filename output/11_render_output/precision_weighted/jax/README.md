# JAX Rendering Results

Generated from GNN POMDP Model: **Precision-Weighted Active Inference Agent**

## Model Information

- **Model Name**: Precision-Weighted Active Inference Agent
- **Model Description**: An Active Inference agent with explicit precision parameters:
- ω (omega): sensory precision weighting likelihood confidence
- γ (gamma): policy precision controlling action randomness
- β (beta): inverse temperature for policy selection (softmax)
- 3 hidden states, 3 observations, 3 actions (same topology as base POMDP)
- Precision parameters enable modeling of attention and confidence
- **Generation Date**: 2026-09-08 06:58:14

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 3

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 3×3 - Maps hidden states to observations
- **B Matrix (Transition)**: 3×3×3 - State transitions given actions
- **C Vector (Preferences)**: Length 3 - Preferences over observations
- **D Vector (Prior)**: Length 3 - Prior beliefs over states
- **E Vector (Habits)**: Length 3 - Policy priors


## Generated Files

- `Precision-Weighted_Active_Inference_Agent_jax.py` - jax simulation script


## Usage

Refer to the main jax documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: jax
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
