"""
GNN Model: Simple MDP Agent
This model describes a fully observable Markov Decision Process (MDP):

- 4 hidden states representing grid positions (corners of a 2x2 grid).
- Observations are identical to states (A = identity matrix).
- 4 actions: stay, move-north, move-south, move-east.
- Preferences strongly favor state/observation 3 (goal location).
- Tests the degenerate POMDP case where partial observability is absent.
Generated: 2026-04-15T12:25:53.937761
"""

import numpy as np
from typing import Dict, List, Any

class SimpleMDPAgentModel:
    """GNN Model: Simple MDP Agent"""

    def __init__(self):
        self.model_name = "Simple MDP Agent"
        self.version = "1.0"
        self.annotation = "This model describes a fully observable Markov Decision Process (MDP):

- 4 hidden states representing grid positions (corners of a 2x2 grid).
- Observations are identical to states (A = identity matrix).
- 4 actions: stay, move-north, move-south, move-east.
- Preferences strongly favor state/observation 3 (goal location).
- Tests the degenerate POMDP case where partial observability is absent."

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Observation model: identity (fully observable)",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [4, 4, 4],
                "description": "State transitions given state and action",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [4],
                "description": "Log-preferences over observations",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [4],
                "description": "Prior over initial hidden states",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Expected Free Energy (per policy)",
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Current observation (same as state in MDP)",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Current hidden state distribution",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Next hidden state distribution",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time step",
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Action taken",
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4],
                "description": "Policy (distribution over actions)",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "B": [[[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9]], [[0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1]], [[0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9], [0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]], [[0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1], [0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]],
            "C": [[0.0, 0.0, 0.0, 3.0]],
            "D": [[0.25, 0.25, 0.25, 0.25]],
        }

# MODEL_DATA: {"model_name":"Simple MDP Agent","annotation":"This model describes a fully observable Markov Decision Process (MDP):\n\n- 4 hidden states representing grid positions (corners of a 2x2 grid).\n- Observations are identical to states (A = identity matrix).\n- 4 actions: stay, move-north, move-south, move-east.\n- Preferences strongly favor state/observation 3 (goal location).\n- Tests the degenerate POMDP case where partial observability is absent.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[4,4]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[4,4,4]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[4]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[4]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.1,0.0,0.0],[0.1,0.9,0.0,0.0],[0.0,0.0,0.9,0.1],[0.0,0.0,0.1,0.9]],[[0.1,0.9,0.0,0.0],[0.9,0.1,0.0,0.0],[0.0,0.0,0.1,0.9],[0.0,0.0,0.9,0.1]],[[0.0,0.0,0.9,0.1],[0.0,0.0,0.1,0.9],[0.9,0.1,0.0,0.0],[0.1,0.9,0.0,0.0]],[[0.0,0.0,0.1,0.9],[0.0,0.0,0.9,0.1],[0.1,0.9,0.0,0.0],[0.9,0.1,0.0,0.0]]],"param_type":"constant"},{"name":"C","value":[[0.0,0.0,0.0,3.0]],"param_type":"constant"},{"name":"D","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
