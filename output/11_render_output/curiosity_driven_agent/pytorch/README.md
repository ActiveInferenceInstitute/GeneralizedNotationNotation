# PYTORCH Rendering Results

Generated from GNN POMDP Model: **Curiosity-Driven Active Inference Agent**

## Model Information

- **Model Name**: Curiosity-Driven Active Inference Agent
- **Model Description**: An Active Inference agent with:
- Explicit epistemic value (information gain / Bayesian surprise) component in G
- Separate instrumental value (preference satisfaction) component
- Precision parameter γ weighting epistemic vs instrumental contributions
- 5 hidden states, 5 observations, 4 actions in a navigation context
- Agent is rewarded for reducing posterior uncertainty
- **Generation Date**: 2026-09-08 06:58:14

## POMDP Dimensions

- **Number of States**: 5
- **Number of Observations**: 5
- **Number of Actions**: 4

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 5×5 - Maps hidden states to observations
- **B Matrix (Transition)**: 5×5×4 - State transitions given actions
- **C Vector (Preferences)**: Length 5 - Preferences over observations
- **D Vector (Prior)**: Length 5 - Prior beliefs over states
- **E Vector (Habits)**: Length 4 - Policy priors


## Generated Files

- `Curiosity-Driven_Active_Inference_Agent_pytorch.py` - pytorch simulation script


## Usage

Refer to the main pytorch documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pytorch
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
