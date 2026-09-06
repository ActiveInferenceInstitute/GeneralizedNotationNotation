# PYTORCH Rendering Results

Generated from GNN POMDP Model: **Factorized Posterior Agent**

## Model Information

- **Model Name**: Factorized Posterior Agent
- **Model Description**: A mean-field factorized POMDP agent. The joint posterior over two
independent state factors `s_1` (location) and `s_2` (goal identity) is
approximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).
This is the canonical simplification used in variational inference when
exact joint posteriors are computationally intractable.
- Two state factors: location (4 states), goal (2 states)
- Two observation modalities: visual (3 obs), proprioceptive (2 obs)
- Separate transition matrices B_1 (location × action) and B_2 (goal is static)
- Explicit factorization declared in ## Equations
- Tests multi-factor / multi-modality handling in the parser
- **Generation Date**: 2026-09-05 20:25:29

## POMDP Dimensions

- **Number of States**: 8
- **Number of Observations**: 6
- **Number of Actions**: 3

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `Factorized_Posterior_Agent_pytorch.py` - pytorch simulation script


## Usage

Refer to the main pytorch documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pytorch
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
