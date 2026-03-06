# DISCOPY Rendering Results

Generated from GNN POMDP Model: **Two State Bistable POMDP**

## Model Information

- **Model Name**: Two State Bistable POMDP
- **Model Description**: This model describes a minimal 2-state bistable POMDP:
- 2 hidden states: "left" and "right" in a symmetric bistable potential.
- 2 noisy observations: the agent gets a noisy readout of which side it is on.
- 2 actions: push-left or push-right.
- The agent prefers observation 1 ("right") over observation 0 ("left").
- Tests the absolute smallest POMDP with full active inference structure.
- **Generation Date**: 2026-03-06 09:48:42

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
- **E Vector (Habits)**: Length 2 - Policy priors


## Generated Files

- `Two State Bistable POMDP_discopy.py` - discopy simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main discopy documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: discopy
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
