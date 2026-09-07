# PYMDP Rendering Results

Generated from GNN POMDP Model: **Three-Level Temporal Hierarchy Agent**

## Model Information

- **Model Name**: Three-Level Temporal Hierarchy Agent
- **Model Description**: A three-level hierarchical Active Inference agent with distinct temporal scales:
- Level 0 (fast, 100ms): Sensorimotor control — immediate reflexive responses
- Level 1 (medium, 1s): Tactical planning — goal-directed behavior sequences
- Level 2 (slow, 10s): Strategic planning — long-term objective management
- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences
- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy
- Each level maintains its own generative model with A, B, C, D matrices
- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)
- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference
- **Generation Date**: 2026-09-06 11:18:26

## POMDP Dimensions

- **Number of States**: 24
- **Number of Observations**: 36
- **Number of Actions**: 3

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `Three-Level_Temporal_Hierarchy_Agent_pymdp.py` - pymdp simulation script


## Usage

Refer to the main pymdp documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: pymdp
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
