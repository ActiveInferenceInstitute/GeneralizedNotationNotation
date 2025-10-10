"""
GNN Model: Active Inference Chronic Pain Multi-Theory Model v1
This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:

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
Generated: 2025-10-10T10:34:18.497669
"""

import numpy as np
from typing import Dict, List, Any

class ActiveInferenceChronicPainMultiTheoryModelv1Model:
    """GNN Model: Active Inference Chronic Pain Multi-Theory Model v1"""

    def __init__(self):
        self.model_name = "Active Inference Chronic Pain Multi-Theory Model v1"
        self.version = "1.0"
        self.annotation = "This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:

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
- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [72, 378],
                "description": "72 observations x 378 hidden state combinations",
            },
            "Attn": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Attention allocation (3 actions: distraction, monitoring, catastrophizing)",
            },
            "Auto": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Autonomic response (2 levels: normal, hyperarousal)",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [378, 378, 81],
                "description": "State transitions given previous state and action",
            },
            "Behav": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Behavioral strategy (3 actions: avoidance, pacing, engagement)",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [72],
                "description": "Log-preferences over pain-related observations",
            },
            "C_sens": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Central sensitization (2 levels: absent, present)",
            },
            "Cog": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [7, 1],
                "description": "Cognitive-affective state (7 levels: adaptive, vigilant, fearful, catastrophizing, depressed, anxious, alexithymic)",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [378],
                "description": "Prior over initial hidden states (acute vs chronic pain)",
            },
            "D_mod": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Descending modulation (3 levels: facilitation, neutral, inhibition)",
            },
            "Desc_C": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Descending control (3 actions: low, moderate, high endogenous analgesia)",
            },
            "E": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [81],
                "description": "Initial policy prior over pain coping strategies",
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Variational Free Energy for state inference",
            },
            "Func": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Functional capacity (3 levels: full, limited, disabled)",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Spinal gate state (3 levels: open, modulated, closed)",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Expected Free Energy (per policy)",
            },
            "P_sens": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Peripheral sensitization (3 levels: normal, moderate, severe)",
            },
            "Pain_I": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Pain intensity (4 levels: none, mild, moderate, severe)",
            },
            "Pain_Q": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Pain quality (3 levels: nociceptive, neuropathic, nociplastic)",
            },
            "Reapp": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Cognitive reappraisal (3 actions: negative, neutral, positive)",
            },
            "T": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Tissue state (3 levels: healed, inflamed, damaged)",
            },
            "t_fast": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Fast timescale (milliseconds to seconds: neural responses, acute signaling)",
            },
            "t_medium": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Medium timescale (minutes to hours: cognitive-affective, behavioral adaptation)",
            },
            "t_slow": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Slow timescale (hours to days: tissue healing, sensitization, chronic changes)",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.7, 0.25, 0.05, 0.0, 0.8, 0.15, 0.05, 0.9, 0.08, 0.02, 0.85, 0.15], [0.6, 0.3, 0.1, 0.0, 0.75, 0.2, 0.05, 0.85, 0.1, 0.05, 0.8, 0.2], [0.45, 0.35, 0.15, 0.05, 0.7, 0.2, 0.1, 0.75, 0.15, 0.1, 0.7, 0.3], [0.2, 0.3, 0.35, 0.15, 0.5, 0.3, 0.2, 0.5, 0.3, 0.2, 0.4, 0.6], [0.3, 0.4, 0.25, 0.05, 0.6, 0.25, 0.15, 0.6, 0.25, 0.15, 0.5, 0.5], [0.4, 0.35, 0.2, 0.05, 0.65, 0.25, 0.1, 0.7, 0.2, 0.1, 0.6, 0.4], [0.55, 0.3, 0.12, 0.03, 0.7, 0.22, 0.08, 0.8, 0.15, 0.05, 0.75, 0.25], [0.8, 0.15, 0.05, 0.0, 0.85, 0.12, 0.03, 0.95, 0.04, 0.01, 0.9, 0.1], [0.75, 0.2, 0.05, 0.0, 0.8, 0.15, 0.05, 0.9, 0.08, 0.02, 0.85, 0.15], [0.65, 0.25, 0.08, 0.02, 0.75, 0.18, 0.07, 0.85, 0.12, 0.03, 0.8, 0.2], [0.9, 0.08, 0.02, 0.0, 0.9, 0.08, 0.02, 0.98, 0.02, 0.0, 0.95, 0.05], [0.85, 0.12, 0.03, 0.0, 0.88, 0.1, 0.02, 0.95, 0.04, 0.01, 0.92, 0.08], [0.05, 0.15, 0.45, 0.35, 0.3, 0.4, 0.3, 0.2, 0.4, 0.4, 0.25, 0.75], [0.02, 0.1, 0.4, 0.48, 0.2, 0.35, 0.45, 0.1, 0.35, 0.55, 0.15, 0.85], [0.35, 0.4, 0.2, 0.05, 0.7, 0.25, 0.05, 0.7, 0.2, 0.1, 0.75, 0.25], [0.2, 0.35, 0.3, 0.15, 0.55, 0.3, 0.15, 0.55, 0.3, 0.15, 0.6, 0.4], [0.15, 0.3, 0.4, 0.15, 0.4, 0.35, 0.25, 0.4, 0.35, 0.25, 0.45, 0.55], [0.05, 0.2, 0.45, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.35, 0.65], [0.4, 0.4, 0.15, 0.05, 0.75, 0.2, 0.05, 0.8, 0.15, 0.05, 0.8, 0.2], [0.25, 0.4, 0.25, 0.1, 0.65, 0.25, 0.1, 0.65, 0.25, 0.1, 0.7, 0.3], [0.2, 0.35, 0.3, 0.15, 0.55, 0.3, 0.15, 0.5, 0.3, 0.2, 0.6, 0.4], [0.08, 0.25, 0.4, 0.27, 0.4, 0.35, 0.25, 0.35, 0.4, 0.25, 0.45, 0.55], [0.05, 0.2, 0.4, 0.35, 0.3, 0.4, 0.3, 0.25, 0.45, 0.3, 0.35, 0.65], [0.02, 0.1, 0.35, 0.53, 0.2, 0.35, 0.45, 0.15, 0.45, 0.4, 0.2, 0.8], [0.02, 0.08, 0.3, 0.6, 0.15, 0.3, 0.55, 0.1, 0.4, 0.5, 0.15, 0.85], [0.1, 0.3, 0.4, 0.2, 0.5, 0.35, 0.15, 0.45, 0.35, 0.2, 0.5, 0.5], [0.05, 0.25, 0.45, 0.25, 0.4, 0.4, 0.2, 0.4, 0.4, 0.2, 0.45, 0.55], [0.03, 0.15, 0.42, 0.4, 0.3, 0.4, 0.3, 0.3, 0.45, 0.25, 0.35, 0.65], [0.01, 0.08, 0.35, 0.56, 0.2, 0.35, 0.45, 0.2, 0.45, 0.35, 0.25, 0.75], [0.01, 0.05, 0.3, 0.64, 0.15, 0.3, 0.55, 0.15, 0.5, 0.35, 0.2, 0.8], [0.0, 0.02, 0.2, 0.78, 0.1, 0.25, 0.65, 0.08, 0.45, 0.47, 0.1, 0.9], [0.7, 0.25, 0.05, 0.0, 0.8, 0.15, 0.05, 0.8, 0.15, 0.05, 0.85, 0.15], [0.55, 0.35, 0.08, 0.02, 0.75, 0.2, 0.05, 0.75, 0.2, 0.05, 0.8, 0.2], [0.15, 0.3, 0.4, 0.15, 0.45, 0.35, 0.2, 0.5, 0.35, 0.15, 0.55, 0.45], [0.05, 0.2, 0.45, 0.3, 0.3, 0.4, 0.3, 0.35, 0.4, 0.25, 0.4, 0.6]],
            "B": [],
            "C": [[2.2, 1.4, 0.5, 0.7, -0.1, -0.8, -0.5, -1.3, -2.2, -1.7, -2.5, -3.4, 2.0, 1.2, 0.3, 0.5, -0.3, -1.0, -0.7, -1.5, -2.4, -1.9, -2.7, -3.6, 1.6, 0.8, -0.1, 0.1, -0.7, -1.4, -1.1, -1.9, -2.8, -2.3, -3.1, -4.0, 1.0, 0.2, -0.7, -0.5, -1.3, -2.0, -1.7, -2.5, -3.4, -2.9, -3.7, -4.6, 0.5, -0.3, -1.2, -1.0, -1.8, -2.5, -2.2, -3.0, -3.9, -3.4, -4.2, -5.1, -0.2, -1.0, -1.9, -1.7, -2.5, -3.2, -2.9, -3.7, -4.6, -4.1, -4.9, -5.8]],
            "D": [[0.1, 0.5, 0.4, 0.6, 0.3, 0.1, 0.4, 0.4, 0.2, 0.9, 0.1, 0.2, 0.6, 0.2, 0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]],
            "E": [[0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015, 0.03, 0.045, 0.03, 0.05, 0.075, 0.05, 0.03, 0.045, 0.03, 0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015, 0.01, 0.015, 0.01, 0.017, 0.025, 0.017, 0.01, 0.015, 0.01, 0.02, 0.03, 0.02, 0.033, 0.05, 0.033, 0.02, 0.03, 0.02, 0.01, 0.015, 0.01, 0.017, 0.025, 0.017, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006, 0.012, 0.018, 0.012, 0.02, 0.03, 0.02, 0.012, 0.018, 0.012, 0.006, 0.009, 0.006, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006]],
            "delta_medium_slow: 0.01        # δ": 'τ_medium / τ_slow ≈ 10^-2\nwindow_fast_to_medium: 300     # s (5 min) - Averaging window for fast→medium coupling\nwindow_medium_to_slow: 14400   # s (4 hours) - Averaging window for medium→slow coupling\npopulation_type: "acute"       # Options: "acute", "chronic", "high_risk", "resilient"',
            "epsilon_fast_medium: 0.001     # ε": 'τ_fast / τ_medium ≈ 10^-3',
        }

# MODEL_DATA: {"model_name":"Active Inference Chronic Pain Multi-Theory Model v1","annotation":"This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:\n\n**Multi-Theory Integration:**\n- Peripheral Sensitization: Enhanced nociceptor responsiveness and reduced thresholds (slow timescale)\n- Central Sensitization: Amplified CNS processing and reduced inhibition (slow timescale, one-way process)\n- Gate Control Theory: Spinal modulation of ascending pain signals (fast timescale)\n- Neuromatrix Theory: Distributed network generating pain experience (fast-medium coupling)\n- Predictive Coding: Pain as precision-weighted prediction error (all timescales)\n- Biopsychosocial Integration: Cognitive, emotional, and behavioral factors (medium timescale)\n\n**Three Nested Timescales:**\n1. Fast (ms-s): Neural signaling, gate control, descending modulation, acute pain perception\n2. Medium (min-hrs): Cognitive-affective processes, behavioral strategies, functional capacity\n3. Slow (hrs-days): Tissue healing, peripheral/central sensitization, chronic adaptations\n\n**State Space Structure:**\n- Six hidden state factors (378 combinations): tissue state (slow), peripheral sensitivity (slow), spinal gate (fast), central sensitization (slow), descending modulation (fast), cognitive-affective state (medium)\n- Four observation modalities (72 outcomes): pain intensity (fast), pain quality (fast), functional capacity (medium), autonomic response (fast)\n- Four control factors (81 actions): attention allocation (medium), behavioral strategy (medium), cognitive reappraisal (medium), descending control (fast)\n\n**Key Features:**\n- Timescale separation: \u03b5 (fast/medium) \u2248 10^-3, \u03b4 (medium/slow) \u2248 10^-2\n- Cross-timescale coupling: slow states modulate fast dynamics, fast observations (averaged) drive medium cognition, medium behaviors (averaged) influence slow healing\n- Testable predictions about pain chronification pathways across multiple timescales\n- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[72,378]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[378,378,81]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[72]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[378]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[81]},{"name":"T","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"P_sens","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[3,1]},{"name":"C_sens","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"D_mod","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Cog","var_type":"hidden_state","data_type":"float","dimensions":[7,1]},{"name":"Pain_I","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"Pain_Q","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Func","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Auto","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"Attn","var_type":"action","data_type":"float","dimensions":[3,1]},{"name":"Behav","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Reapp","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Desc_C","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t_fast","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t_medium","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t_slow","var_type":"hidden_state","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["T"],"connection_type":"directed"},{"source_variables":["D"],"target_variables":["P_sens"],"connection_type":"directed"},{"source_variables":["D"],"target_variables":["C_sens"],"connection_type":"directed"},{"source_variables":["D"],"target_variables":["Cog"],"connection_type":"directed"},{"source_variables":["T"],"target_variables":["P_sens"],"connection_type":"directed"},{"source_variables":["P_sens"],"target_variables":["B"],"connection_type":"directed"},{"source_variables":["P_sens"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["D_mod"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["B"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["C_sens"],"connection_type":"directed"},{"source_variables":["C_sens"],"target_variables":["B"],"connection_type":"directed"},{"source_variables":["T"],"target_variables":["C_sens"],"connection_type":"directed"},{"source_variables":["Cog"],"target_variables":["D_mod"],"connection_type":"directed"},{"source_variables":["D_mod"],"target_variables":["B"],"connection_type":"directed"},{"source_variables":["Pain_I"],"target_variables":["Cog"],"connection_type":"directed"},{"source_variables":["Func"],"target_variables":["Cog"],"connection_type":"directed"},{"source_variables":["Cog"],"target_variables":["B"],"connection_type":"directed"},{"source_variables":["T"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["P_sens"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["C_sens"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["D_mod"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["Cog"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["Attn"],"target_variables":["Cog"],"connection_type":"directed"},{"source_variables":["Behav"],"target_variables":["T"],"connection_type":"directed"},{"source_variables":["Behav"],"target_variables":["Func"],"connection_type":"directed"},{"source_variables":["Reapp"],"target_variables":["Cog"],"connection_type":"directed"},{"source_variables":["Desc_C"],"target_variables":["D_mod"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["Attn"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["Behav"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["Attn"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["Behav"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["Reapp"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["Desc_C"],"connection_type":"directed"},{"source_variables":["Attn"],"target_variables":["Behav"],"connection_type":"directed"},{"source_variables":["Behav"],"target_variables":["Reapp"],"connection_type":"directed"},{"source_variables":["Reapp"],"target_variables":["Desc_C"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.7,0.25,0.05,0.0,0.8,0.15,0.05,0.9,0.08,0.02,0.85,0.15],[0.6,0.3,0.1,0.0,0.75,0.2,0.05,0.85,0.1,0.05,0.8,0.2],[0.45,0.35,0.15,0.05,0.7,0.2,0.1,0.75,0.15,0.1,0.7,0.3],[0.2,0.3,0.35,0.15,0.5,0.3,0.2,0.5,0.3,0.2,0.4,0.6],[0.3,0.4,0.25,0.05,0.6,0.25,0.15,0.6,0.25,0.15,0.5,0.5],[0.4,0.35,0.2,0.05,0.65,0.25,0.1,0.7,0.2,0.1,0.6,0.4],[0.55,0.3,0.12,0.03,0.7,0.22,0.08,0.8,0.15,0.05,0.75,0.25],[0.8,0.15,0.05,0.0,0.85,0.12,0.03,0.95,0.04,0.01,0.9,0.1],[0.75,0.2,0.05,0.0,0.8,0.15,0.05,0.9,0.08,0.02,0.85,0.15],[0.65,0.25,0.08,0.02,0.75,0.18,0.07,0.85,0.12,0.03,0.8,0.2],[0.9,0.08,0.02,0.0,0.9,0.08,0.02,0.98,0.02,0.0,0.95,0.05],[0.85,0.12,0.03,0.0,0.88,0.1,0.02,0.95,0.04,0.01,0.92,0.08],[0.05,0.15,0.45,0.35,0.3,0.4,0.3,0.2,0.4,0.4,0.25,0.75],[0.02,0.1,0.4,0.48,0.2,0.35,0.45,0.1,0.35,0.55,0.15,0.85],[0.35,0.4,0.2,0.05,0.7,0.25,0.05,0.7,0.2,0.1,0.75,0.25],[0.2,0.35,0.3,0.15,0.55,0.3,0.15,0.55,0.3,0.15,0.6,0.4],[0.15,0.3,0.4,0.15,0.4,0.35,0.25,0.4,0.35,0.25,0.45,0.55],[0.05,0.2,0.45,0.3,0.3,0.4,0.3,0.3,0.4,0.3,0.35,0.65],[0.4,0.4,0.15,0.05,0.75,0.2,0.05,0.8,0.15,0.05,0.8,0.2],[0.25,0.4,0.25,0.1,0.65,0.25,0.1,0.65,0.25,0.1,0.7,0.3],[0.2,0.35,0.3,0.15,0.55,0.3,0.15,0.5,0.3,0.2,0.6,0.4],[0.08,0.25,0.4,0.27,0.4,0.35,0.25,0.35,0.4,0.25,0.45,0.55],[0.05,0.2,0.4,0.35,0.3,0.4,0.3,0.25,0.45,0.3,0.35,0.65],[0.02,0.1,0.35,0.53,0.2,0.35,0.45,0.15,0.45,0.4,0.2,0.8],[0.02,0.08,0.3,0.6,0.15,0.3,0.55,0.1,0.4,0.5,0.15,0.85],[0.1,0.3,0.4,0.2,0.5,0.35,0.15,0.45,0.35,0.2,0.5,0.5],[0.05,0.25,0.45,0.25,0.4,0.4,0.2,0.4,0.4,0.2,0.45,0.55],[0.03,0.15,0.42,0.4,0.3,0.4,0.3,0.3,0.45,0.25,0.35,0.65],[0.01,0.08,0.35,0.56,0.2,0.35,0.45,0.2,0.45,0.35,0.25,0.75],[0.01,0.05,0.3,0.64,0.15,0.3,0.55,0.15,0.5,0.35,0.2,0.8],[0.0,0.02,0.2,0.78,0.1,0.25,0.65,0.08,0.45,0.47,0.1,0.9],[0.7,0.25,0.05,0.0,0.8,0.15,0.05,0.8,0.15,0.05,0.85,0.15],[0.55,0.35,0.08,0.02,0.75,0.2,0.05,0.75,0.2,0.05,0.8,0.2],[0.15,0.3,0.4,0.15,0.45,0.35,0.2,0.5,0.35,0.15,0.55,0.45],[0.05,0.2,0.45,0.3,0.3,0.4,0.3,0.35,0.4,0.25,0.4,0.6]],"param_type":"constant"},{"name":"B","value":[],"param_type":"constant"},{"name":"C","value":[[2.2,1.4,0.5,0.7,-0.1,-0.8,-0.5,-1.3,-2.2,-1.7,-2.5,-3.4,2.0,1.2,0.3,0.5,-0.3,-1.0,-0.7,-1.5,-2.4,-1.9,-2.7,-3.6,1.6,0.8,-0.1,0.1,-0.7,-1.4,-1.1,-1.9,-2.8,-2.3,-3.1,-4.0,1.0,0.2,-0.7,-0.5,-1.3,-2.0,-1.7,-2.5,-3.4,-2.9,-3.7,-4.6,0.5,-0.3,-1.2,-1.0,-1.8,-2.5,-2.2,-3.0,-3.9,-3.4,-4.2,-5.1,-0.2,-1.0,-1.9,-1.7,-2.5,-3.2,-2.9,-3.7,-4.6,-4.1,-4.9,-5.8]],"param_type":"constant"},{"name":"D","value":[[0.1,0.5,0.4,0.6,0.3,0.1,0.4,0.4,0.2,0.9,0.1,0.2,0.6,0.2,0.3,0.2,0.15,0.1,0.1,0.1,0.05]],"param_type":"constant"},{"name":"E","value":[[0.015,0.023,0.015,0.025,0.038,0.025,0.015,0.023,0.015,0.03,0.045,0.03,0.05,0.075,0.05,0.03,0.045,0.03,0.015,0.023,0.015,0.025,0.038,0.025,0.015,0.023,0.015,0.01,0.015,0.01,0.017,0.025,0.017,0.01,0.015,0.01,0.02,0.03,0.02,0.033,0.05,0.033,0.02,0.03,0.02,0.01,0.015,0.01,0.017,0.025,0.017,0.01,0.015,0.01,0.006,0.009,0.006,0.01,0.015,0.01,0.006,0.009,0.006,0.012,0.018,0.012,0.02,0.03,0.02,0.012,0.018,0.012,0.006,0.009,0.006,0.01,0.015,0.01,0.006,0.009,0.006]],"param_type":"constant"},{"name":"epsilon_fast_medium: 0.001     # \u03b5","value":"\u03c4_fast / \u03c4_medium \u2248 10^-3","param_type":"constant"},{"name":"delta_medium_slow: 0.01        # \u03b4","value":"\u03c4_medium / \u03c4_slow \u2248 10^-2\nwindow_fast_to_medium: 300     # s (5 min) - Averaging window for fast\u2192medium coupling\nwindow_medium_to_slow: 14400   # s (4 hours) - Averaging window for medium\u2192slow coupling\npopulation_type: \"acute\"       # Options: \"acute\", \"chronic\", \"high_risk\", \"resilient\"","param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded # Chronic pain model for longitudinal simulation across multiple timescales","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrices","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"HabitVector","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"T","ontology_term":"TissueState","description":null},{"variable_name":"P_sens","ontology_term":"PeripheralSensitization","description":null},{"variable_name":"G","ontology_term":"SpinalGateState","description":null},{"variable_name":"C_sens","ontology_term":"CentralSensitization","description":null},{"variable_name":"D_mod","ontology_term":"DescendingModulation","description":null},{"variable_name":"Cog","ontology_term":"CognitiveAffectiveState","description":null},{"variable_name":"Pain_I","ontology_term":"PainIntensityObservation","description":null},{"variable_name":"Pain_Q","ontology_term":"PainQualityObservation","description":null},{"variable_name":"Func","ontology_term":"FunctionalCapacityObservation","description":null},{"variable_name":"Auto","ontology_term":"AutonomicResponseObservation","description":null},{"variable_name":"Attn","ontology_term":"AttentionAllocationControl","description":null},{"variable_name":"Behav","ontology_term":"BehavioralStrategyControl","description":null},{"variable_name":"Reapp","ontology_term":"CognitiveReappraisalControl","description":null},{"variable_name":"Desc_C","ontology_term":"DescendingControlAction","description":null},{"variable_name":"t_fast","ontology_term":"FastTimescale","description":null},{"variable_name":"t_medium","ontology_term":"MediumTimescale","description":null},{"variable_name":"t_slow","ontology_term":"SlowTimescale","description":null}]}
