# BNLEARN Rendering Results

Generated from GNN POMDP Model: **Multi Armed Bandit Agent**

## Model Information

- **Model Name**: Multi Armed Bandit Agent
- **Model Description**: This model describes a 3-armed bandit as a degenerate POMDP:
- 3 hidden states representing the "reward context" (which arm is currently best).
- 3 observations representing reward signals (no-reward, small-reward, big-reward).
- 3 actions: pull arm 0, pull arm 1, or pull arm 2.
- Context switches slowly (sticky transitions), testing exploration vs exploitation.
- The agent prefers big-reward observations (observation 2).
- Tests the bandit structure: meaningful actions despite nearly-static state dynamics.
- **Generation Date**: 2026-04-10 10:25:04

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


## Generated Files

- `Multi Armed Bandit Agent_bnlearn.py` - bnlearn simulation script


## Usage

Refer to the main bnlearn documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: bnlearn
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
