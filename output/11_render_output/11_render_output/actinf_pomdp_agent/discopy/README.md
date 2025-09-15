# DISCOPY Rendering Results

Generated from GNN POMDP Model: **Classic Active Inference POMDP Agent v1**

## Model Information

- **Model Name**: Classic Active Inference POMDP Agent v1
- **Model Description**: This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.
- **Generation Date**: 2025-09-15 09:41:48

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 1

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 3×3 - Maps hidden states to observations
- **B Matrix (Transition)**: 9×4×3 - State transitions given actions
- **C Vector (Preferences)**: Length 3 - Preferences over observations
- **D Vector (Prior)**: Length 3 - Prior beliefs over states
- **E Vector (Habits)**: Length 3 - Policy priors


## Generated Files

- `Classic Active Inference POMDP Agent v1_discopy.py` - discopy simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main discopy documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: discopy
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
