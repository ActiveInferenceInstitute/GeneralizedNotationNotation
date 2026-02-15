# PYMDP Rendering Results

Generated from GNN POMDP Model: **Active Inference POMDP Agent**

## Model Information

- **Model Name**: Active Inference POMDP Agent
- **Model Description**: This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.
- **Generation Date**: 2026-02-15 15:32:32

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

- `Active Inference POMDP Agent_pymdp.py` - pymdp simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main pymdp documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pymdp
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
