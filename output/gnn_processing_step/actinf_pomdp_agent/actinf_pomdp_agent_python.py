"""
GNN Model: Classic Active Inference POMDP Agent v1
This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.
Generated: 2025-07-25T16:14:47.869685
"""

import numpy as np
from typing import Dict, List, Any

class ClassicActiveInferencePOMDPAgentv1Model:
    """GNN Model: Classic Active Inference POMDP Agent v1"""

    def __init__(self):
        self.model_name = "Classic Active Inference POMDP Agent v1"
        self.version = "1.0"
        self.annotation = "This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions."

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Likelihood mapping hidden states to observations",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [3, 3, 3],
                "description": "State transitions given previous state and action",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "Log-preferences over observations",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "Prior over initial hidden states",
            },
            "E": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3],
                "description": "Initial policy prior (habit) over actions",
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
                "dimensions": [3, 1],
                "description": "Current observation (integer index)",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Current hidden state distribution",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
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
                "dimensions": [3],
                "description": "Policy (distribution over actions), no planning",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
            "B": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
            "C": [[0.1, 0.1, 1.0]],
            "D": [[0.33333, 0.33333, 0.33333]],
            "E": [[0.33333, 0.33333, 0.33333]],
            "num_actions: 3       # B actions_dim": '3 (controlled by π)',
        }

# MODEL_DATA: {"model_name":"Classic Active Inference POMDP Agent v1","annotation":"This model describes a classic Active Inference agent for a discrete POMDP:\n- One observation modality (\"state_observation\") with 3 possible outcomes.\n- One hidden state factor (\"location\") with 3 possible states.\n- The hidden state is fully controllable via 3 discrete actions.\n- The agent's preferences are encoded as log-probabilities over observations.\n- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,3]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],"param_type":"constant"},{"name":"B","value":[[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]],[[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C","value":[[0.1,0.1,1.0]],"param_type":"constant"},{"name":"D","value":[[0.33333,0.33333,0.33333]],"param_type":"constant"},{"name":"E","value":[[0.33333,0.33333,0.33333]],"param_type":"constant"},{"name":"num_actions: 3       # B actions_dim","value":"3 (controlled by \u03c0)","param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded # The agent is defined for an unbounded time horizon; simulation runs may specify a finite horizon.","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"Habit","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector # Distribution over actions","description":null},{"variable_name":"u","ontology_term":"Action       # Chosen action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
