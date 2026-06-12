# PYMDP Rendering Results

Generated from GNN POMDP Model: **Time-Varying Transition Dynamics Agent**

## Model Information

- **Model Name**: Time-Varying Transition Dynamics Agent
- **Model Description**: A POMDP agent operating in a non-stationary environment. The key feature
is that the transition matrix `B` is indexed by time (`B_t`), capturing
dynamics that evolve across the planning horizon — e.g., shifting wind
patterns for a sailing agent, or changing opponent strategy in a
sequential game.
- 3 hidden states, 3 observations, 2 actions
- B_t: 3D transition tensor per timestep (shape: next_state × current_state × action)
- Agent must adapt belief updates each step to the current B_t
- Exercises time-varying matrix handling in renderers
This sample pushes the language extensions around time-indexed tensors
and tests downstream code generation when matrix literals are
timestep-dependent.
- **Generation Date**: 2026-05-22 06:18:15

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 2

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 3×3 - Maps hidden states to observations
- **C Vector (Preferences)**: Length 3 - Preferences over observations
- **D Vector (Prior)**: Length 3 - Prior beliefs over states


## Generated Files

- `Time-Varying Transition Dynamics Agent_pymdp.py` - pymdp simulation script


## Usage

Refer to the main pymdp documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pymdp
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
