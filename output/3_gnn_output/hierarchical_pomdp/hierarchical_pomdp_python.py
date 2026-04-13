"""
GNN Model: Hierarchical Active Inference POMDP
A two-level hierarchical POMDP where:
- Level 1 (fast): 4 observations, 4 hidden states, 3 actions
- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood
- Higher-level beliefs are updated at a slower timescale
- Top-down predictions constrain bottom-up inference at Level 1
Generated: 2026-04-12T17:23:00.313526
"""

import numpy as np
from typing import Dict, List, Any

class HierarchicalActiveInferencePOMDPModel:
    """GNN Model: Hierarchical Active Inference POMDP"""

    def __init__(self):
        self.model_name = "Hierarchical Active Inference POMDP"
        self.version = "1.0"
        self.annotation = "A two-level hierarchical POMDP where:
- Level 1 (fast): 4 observations, 4 hidden states, 3 actions
- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood
- Higher-level beliefs are updated at a slower timescale
- Top-down predictions constrain bottom-up inference at Level 1"

        # Variables
        self.variables = {
            "A1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Level 1 likelihood: observations x hidden states",
            },
            "A2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 2],
                "description": "Level 2 likelihood: maps context to Level 1 hidden state prior",
            },
            "B1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Level 1 transitions: next x prev x actions",
            },
            "B2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2, 1],
                "description": "Level 2 transitions (context switches)",
            },
            "C1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 1 preferences over observations",
            },
            "C2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Level 2 preferences over context",
            },
            "D1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Level 1 prior over hidden states",
            },
            "D2": {
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
            "o1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Level 1 observations",
            },
            "o2": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 2 observation (= Level 1 hidden state distribution)",
            },
            "s1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 1 hidden state distribution",
            },
            "s1_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Level 1 next hidden state",
            },
            "s2": {
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
            "u1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Level 1 action",
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
            "A1": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],
            "A2": [[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9]],
            "B1": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            "B2": [[[0.9, 0.1], [0.1, 0.9]]],
            "C1": [[0.1, 0.1, 0.1, 1.0]],
            "C2": [[0.1, 1.0]],
            "D1": [[0.25, 0.25, 0.25, 0.25]],
            "D2": [[0.5, 0.5]],
        }

# MODEL_DATA: {"model_name":"Hierarchical Active Inference POMDP","annotation":"A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1","variables":[{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s1_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o2","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t1","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"t2","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["s1"],"target_variables":["s1_prime"],"connection_type":"directed"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"source_variables":["\u03c01"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["u1"],"target_variables":["s1_prime"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["o2"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["D1"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["B2"],"connection_type":"undirected"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["s2"],"connection_type":"directed"}],"parameters":[{"name":"A1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"B1","value":[[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]],[[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C1","value":[[0.1,0.1,0.1,1.0]],"param_type":"constant"},{"name":"D1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A2","value":[[0.9,0.1,0.0,0.0],[0.1,0.9,0.0,0.0],[0.0,0.0,0.9,0.1],[0.0,0.0,0.1,0.9]],"param_type":"constant"},{"name":"B2","value":[[[0.9,0.1],[0.1,0.9]]],"param_type":"constant"},{"name":"C2","value":[[0.1,1.0]],"param_type":"constant"},{"name":"D2","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s1","ontology_term":"HiddenState","description":null},{"variable_name":"o1","ontology_term":"Observation","description":null},{"variable_name":"\u03c01","ontology_term":"PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Action","description":null},{"variable_name":"G1","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"HigherLevelLikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"ContextTransitionMatrix","description":null},{"variable_name":"s2","ontology_term":"ContextualHiddenState","description":null},{"variable_name":"o2","ontology_term":"HigherLevelObservation","description":null},{"variable_name":"G2","ontology_term":"HigherLevelExpectedFreeEnergy","description":null}]}
