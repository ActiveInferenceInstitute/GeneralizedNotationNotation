# DISCOPY Rendering Results

Generated from GNN POMDP Model: **Active Inference Neural Response Model v1**

## Model Information

- **Model Name**: Active Inference Neural Response Model v1
- **Model Description**: This model describes how a neuron responds to stimuli using Active Inference principles:
- One primary observation modality (firing_rate) with 4 possible activity levels
- Two auxiliary observation modalities (postsynaptic_potential, calcium_signal) for comprehensive monitoring
- Five hidden state factors representing different aspects of neural computation
- Three control factors for plasticity, channel modulation, and metabolic allocation
- The model captures key neural phenomena: membrane potential dynamics, synaptic plasticity (STDP-like), activity-dependent adaptation, homeostatic regulation, and metabolic constraints
- Preferences encode biologically realistic goals: stable firing rates, energy efficiency, and synaptic balance
- **Generation Date**: 2025-10-09 07:55:54

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 3

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 3×12 - Maps hidden states to observations
- **C Vector (Preferences)**: Length 12 - Preferences over observations
- **D Vector (Prior)**: Length 1 - Prior beliefs over states
- **E Vector (Habits)**: Length 27 - Policy priors


## Generated Files

- `Active Inference Neural Response Model v1_discopy.py` - discopy simulation script


## Warnings

- ⚠️ No initial parameterization found - using defaults


## Usage

Refer to the main discopy documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: discopy
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
