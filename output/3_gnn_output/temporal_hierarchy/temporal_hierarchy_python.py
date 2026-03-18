"""
GNN Model: Three-Level Temporal Hierarchy Agent
A three-level hierarchical Active Inference agent with distinct temporal scales:

- Level 0 (fast, 100ms): Sensorimotor control — immediate reflexive responses
- Level 1 (medium, 1s): Tactical planning — goal-directed behavior sequences
- Level 2 (slow, 10s): Strategic planning — long-term objective management
- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences
- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy
- Each level maintains its own generative model with A, B, C, D matrices
- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)
- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference
Generated: 2026-03-18T10:10:54.789580
"""

import numpy as np
from typing import Dict, List, Any

class ThreeLevelTemporalHierarchyAgentModel:
    """GNN Model: Three-Level Temporal Hierarchy Agent"""

    def __init__(self):
        self.model_name = "Three-Level Temporal Hierarchy Agent"
        self.version = "1.0"
        self.annotation = "A three-level hierarchical Active Inference agent with distinct temporal scales:

- Level 0 (fast, 100ms): Sensorimotor control — immediate reflexive responses
- Level 1 (medium, 1s): Tactical planning — goal-directed behavior sequences
- Level 2 (slow, 10s): Strategic planning — long-term objective management
- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences
- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy
- Each level maintains its own generative model with A, B, C, D matrices
- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)
- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference"

        # Variables
        self.variables = {
            "A0": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3, 4],
                "description": "Level 0 likelihood: P(fast_obs | fast_state)",
            },
            "A1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 3],
                "description": "Level 1 likelihood: P(tactic_obs | tactic_state)",
            },
            "A2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3, 2],
                "description": "Level 2 likelihood: P(strategy_obs | strategy_state)",
            },
            "B0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Level 0 transitions: P(fast_state' | fast_state, fast_action)",
            },
            "B1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 3, 3],
                "description": "Level 1 transitions",
            },
            "B2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2, 2],
                "description": "Level 2 transitions",
            },
            "C0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 0 preferences (modulated by Level 1)",
            },
            "C1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 1 preferences (modulated by Level 2)",
            },
            "C2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 2 preferences (fixed strategic goals)",
            },
            "D0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 0 prior over initial states",
            },
            "D1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 1 prior (modulated by Level 2 predictions)",
            },
            "D2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Level 2 prior over strategies",
            },
            "G0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 0 Expected Free Energy",
            },
            "G1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 1 Expected Free Energy",
            },
            "G2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 2 Expected Free Energy",
            },
            "o0": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [3, 1],
                "description": "Level 0 observation",
            },
            "o1": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 1 observation (= summary of Level 0 state trajectory)",
            },
            "o2": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Level 2 observation (= summary of Level 1 outcomes)",
            },
            "pi0": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 0 policy",
            },
            "pi1": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 1 policy",
            },
            "pi2": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [2],
                "description": "Level 2 policy",
            },
            "s0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 0 hidden state belief",
            },
            "s1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Level 1 hidden state belief",
            },
            "s2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Level 2 hidden state belief",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Global discrete time counter",
            },
            "tau0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 0 time constant (0.1s)",
            },
            "tau1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 1 time constant (1.0s)",
            },
            "tau2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 2 time constant (10.0s)",
            },
            "u0": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 0 action",
            },
            "u1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 1 action",
            },
            "u2": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 2 action",
            },
        }

        # Parameters
        self.parameters = {
            "A0": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]],
            "A1": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.1]],
            "A2": [[0.9, 0.1], [0.1, 0.9], [0.1, 0.1]],
            "C0": [[0.0, -1.0, 1.0]],
            "C1": [[-0.5, 1.0, 1.5, -1.0]],
            "C2": [[-1.0, 2.0, 0.5]],
            "D0": [[0.25, 0.25, 0.25, 0.25]],
            "D1": [[0.33, 0.33, 0.34]],
            "D2": [[0.5, 0.5]],
            "tau0": [[0.1]],
            "tau1": [[1.0]],
            "tau2": [[10.0]],
        }

# MODEL_DATA: {"model_name":"Three-Level Temporal Hierarchy Agent","annotation":"A three-level hierarchical Active Inference agent with distinct temporal scales:\n\n- Level 0 (fast, 100ms): Sensorimotor control \u2014 immediate reflexive responses\n- Level 1 (medium, 1s): Tactical planning \u2014 goal-directed behavior sequences\n- Level 2 (slow, 10s): Strategic planning \u2014 long-term objective management\n- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences\n- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy\n- Each level maintains its own generative model with A, B, C, D matrices\n- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)\n- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference","variables":[{"name":"A0","var_type":"action","data_type":"float","dimensions":[3,4]},{"name":"B0","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C0","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D0","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o0","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"pi0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u0","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,3]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[3,3,3]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o1","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[3,2]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,2]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o2","var_type":"observation","data_type":"float","dimensions":[3,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D0"],"target_variables":["s0"],"connection_type":"directed"},{"source_variables":["s0"],"target_variables":["A0"],"connection_type":"undirected"},{"source_variables":["A0"],"target_variables":["o0"],"connection_type":"undirected"},{"source_variables":["C0"],"target_variables":["G0"],"connection_type":"directed"},{"source_variables":["G0"],"target_variables":["pi0"],"connection_type":"directed"},{"source_variables":["pi0"],"target_variables":["u0"],"connection_type":"directed"},{"source_variables":["B0"],"target_variables":["u0"],"connection_type":"directed"},{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"source_variables":["pi1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["o2"],"connection_type":"undirected"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"source_variables":["pi2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["B2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["C1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["C0"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["D1"],"connection_type":"directed"},{"source_variables":["s0"],"target_variables":["o1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["o2"],"connection_type":"directed"}],"parameters":[{"name":"A0","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05]],"param_type":"constant"},{"name":"C0","value":[[0.0,-1.0,1.0]],"param_type":"constant"},{"name":"D0","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A1","value":[[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.1,0.1,0.1]],"param_type":"constant"},{"name":"C1","value":[[-0.5,1.0,1.5,-1.0]],"param_type":"constant"},{"name":"D1","value":[[0.33,0.33,0.34]],"param_type":"constant"},{"name":"A2","value":[[0.9,0.1],[0.1,0.9],[0.1,0.1]],"param_type":"constant"},{"name":"C2","value":[[-1.0,2.0,0.5]],"param_type":"constant"},{"name":"D2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"tau0","value":[[0.1]],"param_type":"constant"},{"name":"tau1","value":[[1.0]],"param_type":"constant"},{"name":"tau2","value":[[10.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":100,"step_size":null},"ontology_mappings":[{"variable_name":"A0","ontology_term":"FastLikelihoodMatrix","description":null},{"variable_name":"B0","ontology_term":"FastTransitionMatrix","description":null},{"variable_name":"C0","ontology_term":"FastPreferenceVector","description":null},{"variable_name":"D0","ontology_term":"FastPrior","description":null},{"variable_name":"s0","ontology_term":"FastHiddenState","description":null},{"variable_name":"o0","ontology_term":"FastObservation","description":null},{"variable_name":"pi0","ontology_term":"FastPolicyVector","description":null},{"variable_name":"u0","ontology_term":"FastAction","description":null},{"variable_name":"G0","ontology_term":"FastExpectedFreeEnergy","description":null},{"variable_name":"A1","ontology_term":"TacticalLikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"TacticalTransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"TacticalPreferenceVector","description":null},{"variable_name":"D1","ontology_term":"TacticalPrior","description":null},{"variable_name":"s1","ontology_term":"TacticalHiddenState","description":null},{"variable_name":"o1","ontology_term":"TacticalObservation","description":null},{"variable_name":"pi1","ontology_term":"TacticalPolicyVector","description":null},{"variable_name":"u1","ontology_term":"TacticalAction","description":null},{"variable_name":"G1","ontology_term":"TacticalExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"StrategicLikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"StrategicTransitionMatrix","description":null},{"variable_name":"C2","ontology_term":"StrategicPreferenceVector","description":null},{"variable_name":"D2","ontology_term":"StrategicPrior","description":null},{"variable_name":"s2","ontology_term":"StrategicHiddenState","description":null},{"variable_name":"o2","ontology_term":"StrategicObservation","description":null},{"variable_name":"pi2","ontology_term":"StrategicPolicyVector","description":null},{"variable_name":"u2","ontology_term":"StrategicAction","description":null},{"variable_name":"G2","ontology_term":"StrategicExpectedFreeEnergy","description":null},{"variable_name":"tau0","ontology_term":"FastTimeConstant","description":null},{"variable_name":"tau1","ontology_term":"TacticalTimeConstant","description":null},{"variable_name":"tau2","ontology_term":"StrategicTimeConstant","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
