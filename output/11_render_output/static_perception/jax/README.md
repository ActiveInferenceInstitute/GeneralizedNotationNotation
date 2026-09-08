# JAX Rendering Results

Generated from GNN POMDP Model: **Static Perception Model**

## Model Information

- **Model Name**: Static Perception Model
- **Model Description**: The simplest Active Inference model demonstrating pure perception:
- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- Minimal 2-action transition component B so the model is a complete POMDP
(renderable and executable by pymdp and the general simulation frameworks)
- Suitable as a minimal baseline and for testing perception-only inference
- **Generation Date**: 2026-09-08 06:56:54

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

- `Static_Perception_Model_jax.py` - jax simulation script


## Usage

Refer to the main jax documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: jax
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
