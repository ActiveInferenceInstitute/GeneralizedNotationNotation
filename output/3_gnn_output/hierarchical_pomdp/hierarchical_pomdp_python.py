"""
GNN Model: Hierarchical Active Inference POMDP
A two-level hierarchical POMDP where:
- Level 1 (fast): 4 observations, 4 hidden states, 3 actions
- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood
- Higher-level beliefs are updated at a slower timescale
- Top-down predictions constrain bottom-up inference at Level 1
Generated: 2026-09-08T06:53:37.492986
"""

import numpy as np
from typing import Dict, List, Any

class HierarchicalActiveInferencePOMDPModel:
    """GNN Model: Hierarchical Active Inference POMDP"""

    def __init__(self):
        self.model_name = "Hierarchical Active Inference POMDP"
        self.version = "1.0"
        self.annotation = "A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1"

        # Variables
        self.variables = {
            "A_level1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Level 1 likelihood: observations x hidden states",
            },
            "A_level2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 2],
                "description": "Level 2 likelihood: maps context to Level 1 hidden state prior",
            },
            "B_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Level 1 transitions: next x prev x actions",
            },
            "B_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2, 1],
                "description": "Level 2 transitions (context switches)",
            },
            "C_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 1 preferences over observations",
            },
            "C_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Level 2 preferences over context",
            },
            "D_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 1 prior over hidden states",
            },
            "D_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Level 2 prior over contextual states",
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
            "o_level1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Level 1 observations",
            },
            "o_level2": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 2 observation (= Level 1 hidden state distribution)",
            },
            "s_level1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 1 hidden state distribution",
            },
            "s_level2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Level 2 contextual hidden state",
            },
            "t1": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Fast timescale counter",
            },
            "t2": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Slow timescale counter",
            },
            "u_level1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 1 action",
            },
            "x_next1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 1 next hidden state",
            },
            "π1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Level 1 policy (actions)",
            },
        }

        # Parameters
        self.parameters = {
            "A_level1": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],
            "A_level2": [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.5, 0.5]],
            "B_level1": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            "B_level2": [[[0.9, 0.1], [0.1, 0.9]]],
            "C_level1": [[0.1, 0.1, 0.1, 1.0]],
            "C_level2": [[0.0, 0.5, 0.0, 0.5]],
            "D_level1": [[0.25, 0.25, 0.25, 0.25]],
            "D_level2": [[0.5, 0.5]],
            "num_actions": 3,
            "num_actions_l1": 3,
            "num_context_states_l2": 2,
            "num_hidden_states": 8,
            "num_hidden_states_l1": 4,
            "num_obs": 16,
            "num_obs_l1": 4,
            "num_timesteps": 20,
            "timescale_ratio": 5,
        }

# MODEL_DATA: {"model_name":"Hierarchical Active Inference POMDP","annotation":"A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1","variables":[{"name":"A_level1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_level1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_level1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_level1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u_level1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level2","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"B_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"C_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"D_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_level2","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t1","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"t2","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_level1"],"target_variables":["s_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["A_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["A_level1"],"target_variables":["o_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c01"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_level1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["o_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level2"],"target_variables":["s_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["A_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level2"],"target_variables":["D_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["B_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["s_level2"],"connection_type":"directed"}],"parameters":[{"name":"A_level1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"B_level1","value":[[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]],[[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C_level1","value":[[0.1,0.1,0.1,1.0]],"param_type":"constant"},{"name":"D_level1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A_level2","value":[[0.9,0.1],[0.1,0.9],[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"B_level2","value":[[[0.9,0.1],[0.1,0.9]]],"param_type":"constant"},{"name":"C_level2","value":[[0.0,0.5,0.0,0.5]],"param_type":"constant"},{"name":"D_level2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":8,"param_type":"constant"},{"name":"num_obs","value":16,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_timesteps","value":20,"param_type":"constant"},{"name":"num_hidden_states_l1","value":4,"param_type":"constant"},{"name":"num_obs_l1","value":4,"param_type":"constant"},{"name":"num_actions_l1","value":3,"param_type":"constant"},{"name":"num_context_states_l2","value":2,"param_type":"constant"},{"name":"timescale_ratio","value":5,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A_level1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_level1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_level1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_level1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_level1","ontology_term":"HiddenState","description":null},{"variable_name":"o_level1","ontology_term":"Observation","description":null},{"variable_name":"\u03c01","ontology_term":"PolicyVector","description":null},{"variable_name":"u_level1","ontology_term":"Action","description":null},{"variable_name":"G1","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"A_level2","ontology_term":"HigherLevelLikelihoodMatrix","description":null},{"variable_name":"B_level2","ontology_term":"ContextTransitionMatrix","description":null},{"variable_name":"s_level2","ontology_term":"ContextualHiddenState","description":null},{"variable_name":"o_level2","ontology_term":"HigherLevelObservation","description":null},{"variable_name":"G2","ontology_term":"HigherLevelExpectedFreeEnergy","description":null}]}
