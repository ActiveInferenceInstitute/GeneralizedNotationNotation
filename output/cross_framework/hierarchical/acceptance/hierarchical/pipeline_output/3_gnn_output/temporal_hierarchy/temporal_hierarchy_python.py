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
Generated: 2026-09-06T11:57:37.923723
"""

import numpy as np
from typing import Dict, List, Any

class ThreeLevelTemporalHierarchyAgentModel:
    """GNN Model: Three-Level Temporal Hierarchy Agent"""

    def __init__(self):
        self.model_name = "Three-Level Temporal Hierarchy Agent"
        self.version = "1.0"
        self.annotation = "A three-level hierarchical Active Inference agent with distinct temporal scales:\n\n- Level 0 (fast, 100ms): Sensorimotor control \u2014 immediate reflexive responses\n- Level 1 (medium, 1s): Tactical planning \u2014 goal-directed behavior sequences\n- Level 2 (slow, 10s): Strategic planning \u2014 long-term objective management\n- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences\n- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy\n- Each level maintains its own generative model with A, B, C, D matrices\n- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)\n- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference"

        # Variables
        self.variables = {
            "A_level0": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3, 4],
                "description": "Level 0 likelihood: P(fast_obs | fast_state)",
            },
            "A_level1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 3],
                "description": "Level 1 likelihood: P(tactic_obs | tactic_state)",
            },
            "A_level2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3, 2],
                "description": "Level 2 likelihood: P(strategy_obs | strategy_state)",
            },
            "B_level0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Level 0 transitions: P(fast_state' | fast_state, fast_action)",
            },
            "B_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 3, 3],
                "description": "Level 1 transitions",
            },
            "B_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2, 2],
                "description": "Level 2 transitions",
            },
            "C_level0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 0 preferences (modulated by Level 1)",
            },
            "C_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 1 preferences (modulated by Level 2)",
            },
            "C_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 2 preferences (fixed strategic goals)",
            },
            "D_level0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 0 prior over initial states",
            },
            "D_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 1 prior (modulated by Level 2 predictions)",
            },
            "D_level2": {
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
            "o_level0": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [3, 1],
                "description": "Level 0 observation",
            },
            "o_level1": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 1 observation (= summary of Level 0 state trajectory)",
            },
            "o_level2": {
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
            "s_level0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 0 hidden state belief",
            },
            "s_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Level 1 hidden state belief",
            },
            "s_level2": {
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
            "tau_level0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 0 time constant (0.1s)",
            },
            "tau_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 1 time constant (1.0s)",
            },
            "tau_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Level 2 time constant (10.0s)",
            },
            "u_level0": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 0 action",
            },
            "u_level1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 1 action",
            },
            "u_level2": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 2 action",
            },
        }

        # Parameters
        self.parameters = {
            "A_level0": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]],
            "A_level1": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.1]],
            "A_level2": [[0.9, 0.1], [0.1, 0.9], [0.1, 0.1]],
            "B_level0": [[[0.9, 0.05, 0.05, 0.0], [0.05, 0.9, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.05, 0.9, 0.05, 0.0], [0.9, 0.05, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.9, 0.05, 0.05, 0.0], [0.05, 0.9, 0.05, 0.0], [0.05, 0.05, 0.9, 0.0], [0.0, 0.0, 0.0, 1.0]]],
            "B_level1": [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]],
            "B_level2": [[[0.95, 0.05], [0.05, 0.95]]],
            "C_level0": [[0.0, -1.0, 1.0]],
            "C_level1": [[-0.5, 1.0, 1.5, -1.0]],
            "C_level2": [[-1.0, 2.0, 0.5]],
            "D_level0": [[0.25, 0.25, 0.25, 0.25]],
            "D_level1": [[0.33, 0.33, 0.34]],
            "D_level2": [[0.5, 0.5]],
            "num_actions": 3,
            "num_actions_l0": 3,
            "num_actions_l1": 3,
            "num_actions_l2": 2,
            "num_hidden_states": 24,
            "num_levels": 3,
            "num_obs": 36,
            "num_obs_l0": 3,
            "num_obs_l1": 4,
            "num_obs_l2": 3,
            "num_states_l0": 4,
            "num_states_l1": 3,
            "num_states_l2": 2,
            "num_timesteps": 100,
            "tau_level0": [[0.1]],
            "tau_level1": [[1.0]],
            "tau_level2": [[10.0]],
            "timescale_ratio_1_0": 10,
            "timescale_ratio_2_1": 10,
        }

# MODEL_DATA: {"model_name":"Three-Level Temporal Hierarchy Agent","annotation":"A three-level hierarchical Active Inference agent with distinct temporal scales:\n\n- Level 0 (fast, 100ms): Sensorimotor control \u2014 immediate reflexive responses\n- Level 1 (medium, 1s): Tactical planning \u2014 goal-directed behavior sequences\n- Level 2 (slow, 10s): Strategic planning \u2014 long-term objective management\n- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences\n- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy\n- Each level maintains its own generative model with A, B, C, D matrices\n- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)\n- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference","variables":[{"name":"A_level0","var_type":"action","data_type":"float","dimensions":[3,4]},{"name":"B_level0","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_level0","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_level0","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_level0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_level0","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"pi0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u_level0","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level1","var_type":"action","data_type":"float","dimensions":[4,3]},{"name":"B_level1","var_type":"hidden_state","data_type":"float","dimensions":[3,3,3]},{"name":"C_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_level1","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"s_level1","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o_level1","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u_level1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level2","var_type":"action","data_type":"float","dimensions":[3,2]},{"name":"B_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,2]},{"name":"C_level2","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_level2","var_type":"observation","data_type":"float","dimensions":[3,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u_level2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau_level0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau_level1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau_level2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_level0"],"target_variables":["s_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level0"],"target_variables":["A_level0"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level0"],"target_variables":["o_level0"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level0"],"target_variables":["G0"],"connection_type":"directed"},{"annotation":null,"source_variables":["G0"],"target_variables":["pi0"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi0"],"target_variables":["u_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level0"],"target_variables":["u_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level1"],"target_variables":["s_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["A_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level1"],"target_variables":["o_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level2"],"target_variables":["s_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["A_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level2"],"target_variables":["o_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi2"],"target_variables":["u_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level2"],"target_variables":["u_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["C_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["C_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["D_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level0"],"target_variables":["o_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["o_level2"],"connection_type":"directed"}],"parameters":[{"name":"A_level0","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05]],"param_type":"constant"},{"name":"C_level0","value":[[0.0,-1.0,1.0]],"param_type":"constant"},{"name":"D_level0","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A_level1","value":[[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.1,0.1,0.1]],"param_type":"constant"},{"name":"C_level1","value":[[-0.5,1.0,1.5,-1.0]],"param_type":"constant"},{"name":"D_level1","value":[[0.33,0.33,0.34]],"param_type":"constant"},{"name":"A_level2","value":[[0.9,0.1],[0.1,0.9],[0.1,0.1]],"param_type":"constant"},{"name":"C_level2","value":[[-1.0,2.0,0.5]],"param_type":"constant"},{"name":"D_level2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"tau_level0","value":[[0.1]],"param_type":"constant"},{"name":"tau_level1","value":[[1.0]],"param_type":"constant"},{"name":"tau_level2","value":[[10.0]],"param_type":"constant"},{"name":"B_level0","value":[[[0.9,0.05,0.05,0.0],[0.05,0.9,0.05,0.0],[0.05,0.05,0.9,0.0],[0.0,0.0,0.0,1.0]],[[0.05,0.9,0.05,0.0],[0.9,0.05,0.05,0.0],[0.05,0.05,0.9,0.0],[0.0,0.0,0.0,1.0]],[[0.9,0.05,0.05,0.0],[0.05,0.9,0.05,0.0],[0.05,0.05,0.9,0.0],[0.0,0.0,0.0,1.0]]],"param_type":"constant"},{"name":"B_level1","value":[[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],[[0.05,0.9,0.05],[0.9,0.05,0.05],[0.05,0.05,0.9]],[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]],"param_type":"constant"},{"name":"B_level2","value":[[[0.95,0.05],[0.05,0.95]]],"param_type":"constant"},{"name":"num_hidden_states","value":24,"param_type":"constant"},{"name":"num_obs","value":36,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_levels","value":3,"param_type":"constant"},{"name":"num_states_l0","value":4,"param_type":"constant"},{"name":"num_obs_l0","value":3,"param_type":"constant"},{"name":"num_actions_l0","value":3,"param_type":"constant"},{"name":"num_states_l1","value":3,"param_type":"constant"},{"name":"num_obs_l1","value":4,"param_type":"constant"},{"name":"num_actions_l1","value":3,"param_type":"constant"},{"name":"num_states_l2","value":2,"param_type":"constant"},{"name":"num_obs_l2","value":3,"param_type":"constant"},{"name":"num_actions_l2","value":2,"param_type":"constant"},{"name":"timescale_ratio_1_0","value":10,"param_type":"constant"},{"name":"timescale_ratio_2_1","value":10,"param_type":"constant"},{"name":"num_timesteps","value":100,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":100,"step_size":null},"ontology_mappings":[{"variable_name":"A_level0","ontology_term":"FastLikelihoodMatrix","description":null},{"variable_name":"B_level0","ontology_term":"FastTransitionMatrix","description":null},{"variable_name":"C_level0","ontology_term":"FastPreferenceVector","description":null},{"variable_name":"D_level0","ontology_term":"FastPrior","description":null},{"variable_name":"s_level0","ontology_term":"FastHiddenState","description":null},{"variable_name":"o_level0","ontology_term":"FastObservation","description":null},{"variable_name":"pi0","ontology_term":"FastPolicyVector","description":null},{"variable_name":"u_level0","ontology_term":"FastAction","description":null},{"variable_name":"G0","ontology_term":"FastExpectedFreeEnergy","description":null},{"variable_name":"A_level1","ontology_term":"TacticalLikelihoodMatrix","description":null},{"variable_name":"B_level1","ontology_term":"TacticalTransitionMatrix","description":null},{"variable_name":"C_level1","ontology_term":"TacticalPreferenceVector","description":null},{"variable_name":"D_level1","ontology_term":"TacticalPrior","description":null},{"variable_name":"s_level1","ontology_term":"TacticalHiddenState","description":null},{"variable_name":"o_level1","ontology_term":"TacticalObservation","description":null},{"variable_name":"pi1","ontology_term":"TacticalPolicyVector","description":null},{"variable_name":"u_level1","ontology_term":"TacticalAction","description":null},{"variable_name":"G1","ontology_term":"TacticalExpectedFreeEnergy","description":null},{"variable_name":"A_level2","ontology_term":"StrategicLikelihoodMatrix","description":null},{"variable_name":"B_level2","ontology_term":"StrategicTransitionMatrix","description":null},{"variable_name":"C_level2","ontology_term":"StrategicPreferenceVector","description":null},{"variable_name":"D_level2","ontology_term":"StrategicPrior","description":null},{"variable_name":"s_level2","ontology_term":"StrategicHiddenState","description":null},{"variable_name":"o_level2","ontology_term":"StrategicObservation","description":null},{"variable_name":"pi2","ontology_term":"StrategicPolicyVector","description":null},{"variable_name":"u_level2","ontology_term":"StrategicAction","description":null},{"variable_name":"G2","ontology_term":"StrategicExpectedFreeEnergy","description":null},{"variable_name":"tau_level0","ontology_term":"FastTimeConstant","description":null},{"variable_name":"tau_level1","ontology_term":"TacticalTimeConstant","description":null},{"variable_name":"tau_level2","ontology_term":"StrategicTimeConstant","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
