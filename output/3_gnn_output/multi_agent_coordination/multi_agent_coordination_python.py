"""
GNN Model: Multi-Agent Cooperative Active Inference
Two Active Inference agents cooperating on a joint task:
- Agent 1 and Agent 2 each maintain independent beliefs
- Shared observation space: agents observe each other's actions
- Joint task state includes both agents' positions (4x4 = 16 joint states)
- Cooperative preferences: both agents prefer the same goal configuration
- Models social cognition and coordination without explicit communication
Generated: 2026-03-03T08:22:02.582124
"""

import numpy as np
from typing import Dict, List, Any

class MultiAgentCooperativeActiveInferenceModel:
    """GNN Model: Multi-Agent Cooperative Active Inference"""

    def __init__(self):
        self.model_name = "Multi-Agent Cooperative Active Inference"
        self.version = "1.0"
        self.annotation = "Two Active Inference agents cooperating on a joint task:
- Agent 1 and Agent 2 each maintain independent beliefs
- Shared observation space: agents observe each other's actions
- Joint task state includes both agents' positions (4x4 = 16 joint states)
- Cooperative preferences: both agents prefer the same goal configuration
- Models social cognition and coordination without explicit communication"

        # Variables
        self.variables = {
            "A1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Agent 1 likelihood",
            },
            "A2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Agent 2 likelihood",
            },
            "B1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Agent 1 transitions (3 actions)",
            },
            "B2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Agent 2 transitions (3 actions)",
            },
            "C1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 preferences",
            },
            "C2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 2 preferences",
            },
            "D1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 prior",
            },
            "D2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 2 prior",
            },
            "G1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Agent 1 EFE",
            },
            "G2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Agent 2 EFE",
            },
            "o1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 1 observations (includes Agent 2 obs)",
            },
            "o2": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 2 observations (includes Agent 1 obs)",
            },
            "o_joint": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Joint observation (goal achievement)",
            },
            "s1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Agent 1 hidden state",
            },
            "s2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Agent 2 hidden state",
            },
            "s_joint": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [16, 1],
                "description": "Joint state (Agent1_pos x Agent2_pos)",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
            },
            "u1": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 1 action",
            },
            "u2": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Agent 2 action",
            },
            "π1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Agent 1 policy",
            },
            "π2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Agent 2 policy",
            },
        }

        # Parameters
        self.parameters = {
            "A1": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],
            "A2": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],
            "B1": [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]],
            "B2": [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]],
            "C1": [[-1.0, -1.0, -1.0, 2.0]],
            "C2": [[-1.0, -1.0, -1.0, 2.0]],
            "D1": [[0.25, 0.25, 0.25, 0.25]],
            "D2": [[0.25, 0.25, 0.25, 0.25]],
        }

# MODEL_DATA: {"model_name":"Multi-Agent Cooperative Active Inference","annotation":"Two Active Inference agents cooperating on a joint task:\n- Agent 1 and Agent 2 each maintain independent beliefs\n- Shared observation space: agents observe each other's actions\n- Joint task state includes both agents' positions (4x4 = 16 joint states)\n- Cooperative preferences: both agents prefer the same goal configuration\n- Models social cognition and coordination without explicit communication","variables":[{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c02","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"s_joint","var_type":"hidden_state","data_type":"float","dimensions":[16,1]},{"name":"o_joint","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["s1"],"target_variables":["s1_prime"],"connection_type":"directed"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"source_variables":["\u03c01"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["o2"],"connection_type":"undirected"},{"source_variables":["s2"],"target_variables":["s2_prime"],"connection_type":"directed"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["\u03c02"],"connection_type":"directed"},{"source_variables":["\u03c02"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["B2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["u1"],"target_variables":["s_joint"],"connection_type":"directed"},{"source_variables":["u2"],"target_variables":["s_joint"],"connection_type":"directed"},{"source_variables":["s_joint"],"target_variables":["o_joint"],"connection_type":"undirected"},{"source_variables":["o1"],"target_variables":["s_joint"],"connection_type":"undirected"},{"source_variables":["o2"],"target_variables":["s_joint"],"connection_type":"undirected"}],"parameters":[{"name":"A1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"A2","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"C1","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"C2","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"D1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"D2","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"B1","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"},{"name":"B2","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":20,"step_size":null},"ontology_mappings":[{"variable_name":"A1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s1","ontology_term":"Agent1HiddenState","description":null},{"variable_name":"o1","ontology_term":"Agent1Observation","description":null},{"variable_name":"\u03c01","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C2","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D2","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s2","ontology_term":"Agent2HiddenState","description":null},{"variable_name":"o2","ontology_term":"Agent2Observation","description":null},{"variable_name":"\u03c02","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"s_joint","ontology_term":"JointState","description":null},{"variable_name":"o_joint","ontology_term":"JointObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
