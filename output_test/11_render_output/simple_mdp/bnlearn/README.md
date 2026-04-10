# BNLEARN Rendering Results

Generated from GNN POMDP Model: **Simple MDP Agent**

## Model Information

- **Model Name**: Simple MDP Agent
- **Model Description**: This model describes a fully observable Markov Decision Process (MDP):
- 4 hidden states representing grid positions (corners of a 2x2 grid).
- Observations are identical to states (A = identity matrix).
- 4 actions: stay, move-north, move-south, move-east.
- Preferences strongly favor state/observation 3 (goal location).
- Tests the degenerate POMDP case where partial observability is absent.
- **Generation Date**: 2026-04-09 14:51:17

## POMDP Dimensions

- **Number of States**: 4
- **Number of Observations**: 4
- **Number of Actions**: 4

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 4×4 - Maps hidden states to observations
- **B Matrix (Transition)**: 4×4×4 - State transitions given actions
- **C Vector (Preferences)**: Length 4 - Preferences over observations
- **D Vector (Prior)**: Length 4 - Prior beliefs over states


## Generated Files

- `Simple MDP Agent_bnlearn.py` - bnlearn simulation script


## Usage

Refer to the main bnlearn documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: bnlearn
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
