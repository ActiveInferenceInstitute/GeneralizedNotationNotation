"""
GNN Model: Multi-Agent Cooperative Active Inference
Two Active Inference agents cooperating on a joint task:

- Agent 1 and Agent 2 each maintain independent beliefs
- Shared observation space: agents observe each other's actions
- Joint task state includes both agents' positions (4x4 = 16 joint states)
- Cooperative preferences: both agents prefer the same goal configuration
- Models social cognition and coordination without explicit communication
Generated: 2026-09-05T20:30:46.301221
"""

import numpy as np
from typing import Dict, List, Any

class MultiAgentCooperativeActiveInferenceModel:
    """GNN Model: Multi-Agent Cooperative Active Inference"""

    def __init__(self):
        self.model_name = "Multi-Agent Cooperative Active Inference"
        self.version = "1.0"
        self.annotation = "Two Active Inference agents cooperating on a joint task:\n\n- Agent 1 and Agent 2 each maintain independent beliefs\n- Shared observation space: agents observe each other's actions\n- Joint task state includes both agents' positions (4x4 = 16 joint states)\n- Cooperative preferences: both agents prefer the same goal configuration\n- Models social cognition and coordination without explicit communication"

        # Variables
        self.variables = {
            "A_agent1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Agent 1 likelihood",
            },
            "A_agent2": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Agent 2 likelihood",
            },
            "B_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Agent 1 transitions (3 actions)",
            },
            "B_agent2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Agent 2 transitions (3 actions)",
            },
            "C_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 preferences",
            },
            "C_agent2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 2 preferences",
            },
            "D_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Agent 1 prior",
            },
            "D_agent2": {
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
            "o_agent1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Agent 1 observations (includes Agent 2 obs)",
            },
            "o_agent2": {
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
            "s_agent1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Agent 1 hidden state",
            },
            "s_agent2": {
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
            "x_next1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Agent 1 next hidden state",
            },
            "x_next2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Agent 2 next hidden state",
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
            "A_agent1": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],
            "A_agent2": [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85]],
            "B_agent1": [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]],
            "B_agent2": [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.0, 0.1], [0.1, 0.0, 0.8, 0.1], [0.0, 0.1, 0.1, 0.8]]],
            "C_agent1": [[-1.0, -1.0, -1.0, 2.0]],
            "C_agent2": [[-1.0, -1.0, -1.0, 2.0]],
            "D_agent1": [[0.25, 0.25, 0.25, 0.25]],
            "D_agent2": [[0.25, 0.25, 0.25, 0.25]],
            "num_actions": 3,
            "num_actions_per_agent": 3,
            "num_agents": 2,
            "num_hidden_states": 16,
            "num_hidden_states_per_agent": 4,
            "num_obs": 16,
            "num_obs_per_agent": 4,
            "num_timesteps": 20,
        }

# MODEL_DATA: {"model_name":"Multi-Agent Cooperative Active Inference","annotation":"Two Active Inference agents cooperating on a joint task:\n\n- Agent 1 and Agent 2 each maintain independent beliefs\n- Shared observation space: agents observe each other's actions\n- Joint task state includes both agents' positions (4x4 = 16 joint states)\n- Cooperative preferences: both agents prefer the same goal configuration\n- Models social cognition and coordination without explicit communication","variables":[{"name":"A_agent1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_agent1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_agent2","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_agent2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c02","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"s_joint","var_type":"hidden_state","data_type":"float","dimensions":[16,1]},{"name":"o_joint","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_agent1"],"target_variables":["s_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent1"],"target_variables":["A_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent1"],"target_variables":["o_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_agent1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["C_agent1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c01"],"target_variables":["u1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent1"],"target_variables":["u1"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_agent2"],"target_variables":["s_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent2"],"target_variables":["A_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent2"],"target_variables":["o_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_agent2"],"target_variables":["x_next2"],"connection_type":"directed"},{"annotation":null,"source_variables":["C_agent2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["\u03c02"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c02"],"target_variables":["u2"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent2"],"target_variables":["u2"],"connection_type":"directed"},{"annotation":null,"source_variables":["u1"],"target_variables":["s_joint"],"connection_type":"directed"},{"annotation":null,"source_variables":["u2"],"target_variables":["s_joint"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_joint"],"target_variables":["o_joint"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o_agent1"],"target_variables":["s_joint"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o_agent2"],"target_variables":["s_joint"],"connection_type":"undirected"}],"parameters":[{"name":"A_agent1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"A_agent2","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"C_agent1","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"C_agent2","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"D_agent1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"D_agent2","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"B_agent1","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"},{"name":"B_agent2","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"},{"name":"num_hidden_states","value":16,"param_type":"constant"},{"name":"num_obs","value":16,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_agents","value":2,"param_type":"constant"},{"name":"num_hidden_states_per_agent","value":4,"param_type":"constant"},{"name":"num_obs_per_agent","value":4,"param_type":"constant"},{"name":"num_actions_per_agent","value":3,"param_type":"constant"},{"name":"num_timesteps","value":20,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":20,"step_size":null},"ontology_mappings":[{"variable_name":"A_agent1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_agent1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_agent1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_agent1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_agent1","ontology_term":"Agent1HiddenState","description":null},{"variable_name":"x_next1","ontology_term":"Agent1NextHiddenState","description":null},{"variable_name":"o_agent1","ontology_term":"Agent1Observation","description":null},{"variable_name":"\u03c01","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A_agent2","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_agent2","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_agent2","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_agent2","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_agent2","ontology_term":"Agent2HiddenState","description":null},{"variable_name":"x_next2","ontology_term":"Agent2NextHiddenState","description":null},{"variable_name":"o_agent2","ontology_term":"Agent2Observation","description":null},{"variable_name":"\u03c02","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"s_joint","ontology_term":"JointState","description":null},{"variable_name":"o_joint","ontology_term":"JointObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
