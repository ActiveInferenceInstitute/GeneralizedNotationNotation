# JAX Rendering Results

Generated from GNN POMDP Model: **Active Inference Chronic Pain Multi-Theory Model v1**

## Model Information

- **Model Name**: Active Inference Chronic Pain Multi-Theory Model v1
- **Model Description**: This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:
**Multi-Theory Integration:**
- Peripheral Sensitization: Enhanced nociceptor responsiveness and reduced thresholds (slow timescale)
- Central Sensitization: Amplified CNS processing and reduced inhibition (slow timescale, one-way process)
- Gate Control Theory: Spinal modulation of ascending pain signals (fast timescale)
- Neuromatrix Theory: Distributed network generating pain experience (fast-medium coupling)
- Predictive Coding: Pain as precision-weighted prediction error (all timescales)
- Biopsychosocial Integration: Cognitive, emotional, and behavioral factors (medium timescale)
**Three Nested Timescales:**
1. Fast (ms-s): Neural signaling, gate control, descending modulation, acute pain perception
2. Medium (min-hrs): Cognitive-affective processes, behavioral strategies, functional capacity
3. Slow (hrs-days): Tissue healing, peripheral/central sensitization, chronic adaptations
**State Space Structure:**
- Six hidden state factors (378 combinations): tissue state (slow), peripheral sensitivity (slow), spinal gate (fast), central sensitization (slow), descending modulation (fast), cognitive-affective state (medium)
- Four observation modalities (72 outcomes): pain intensity (fast), pain quality (fast), functional capacity (medium), autonomic response (fast)
- Four control factors (81 actions): attention allocation (medium), behavioral strategy (medium), cognitive reappraisal (medium), descending control (fast)
**Key Features:**
- Timescale separation: ε (fast/medium) ≈ 10^-3, δ (medium/slow) ≈ 10^-2
- Cross-timescale coupling: slow states modulate fast dynamics, fast observations (averaged) drive medium cognition, medium behaviors (averaged) influence slow healing
- Testable predictions about pain chronification pathways across multiple timescales
- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)
- **Generation Date**: 2025-10-09 07:55:54

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 3

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 38×12 - Maps hidden states to observations
- **C Vector (Preferences)**: Length 72 - Preferences over observations
- **D Vector (Prior)**: Length 2 - Prior beliefs over states
- **E Vector (Habits)**: Length 81 - Policy priors


## Generated Files

- `Active Inference Chronic Pain Multi-Theory Model v1_jax.py` - jax simulation script


## Usage

Refer to the main jax documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: jax
- **File Extension**: .py
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
