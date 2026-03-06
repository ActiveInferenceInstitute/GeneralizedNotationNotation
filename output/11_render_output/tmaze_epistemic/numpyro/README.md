# NUMPYRO Rendering Results

Generated from GNN POMDP Model: **T-Maze Epistemic Foraging Agent**

## Model Information

- **Model Name**: T-Maze Epistemic Foraging Agent
- **Model Description**: The classic T-maze task from Active Inference literature (Friston et al.):
- Agent navigates a T-shaped maze with 4 locations: center, left arm, right arm, cue location
- Two observation modalities: location (where am I?) and reward/cue (what do I see?)
- Reward is hidden behind one of the two arms (left or right), determined by context
- Cue location provides partial information about which arm holds the reward
- Agent must decide: go directly to an arm (exploit) or visit cue location first (explore)
- Demonstrates epistemic foraging: Active Inference naturally balances exploration vs exploitation
- The Expected Free Energy decomposes into epistemic (information gain) + instrumental (reward) value
- **Generation Date**: 2026-03-06 15:00:54

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

- `T-Maze Epistemic Foraging Agent_numpyro.py` - numpyro simulation script


## Usage

Refer to the main numpyro documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: numpyro
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
