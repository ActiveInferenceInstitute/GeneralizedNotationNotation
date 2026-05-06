"""
GNN Model: PyMDP Scaling N2 T50
PyMDP runtime scaling sweep with noisy observation and stochastic transitions.
Generated: 2026-05-06T07:03:55.464121
"""

import numpy as np
from typing import Dict, List, Any

class PyMDPScalingN2T50Model:
    """GNN Model: PyMDP Scaling N2 T50"""

    def __init__(self):
        self.model_name = "PyMDP Scaling N2 T50"
        self.version = "1.0"
        self.annotation = "PyMDP runtime scaling sweep with noisy observation and stochastic transitions."

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [2, 2],
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [2, 2, 2],
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [2],
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2],
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [2, 1],
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [2],
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.95, 0.05], [0.05, 0.95]],
            "B": [[[0.9, 0.1], [0.9, 0.1]], [[0.1, 0.9], [0.1, 0.9]]],
            "C": [[0.0, 3.0]],
            "D": [[0.5, 0.5]],
            "num_actions": 2,
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_timesteps": 50,
        }

# MODEL_DATA: {"model_name":"PyMDP Scaling N2 T50","annotation":"PyMDP runtime scaling sweep with noisy observation and stochastic transitions.","variables":[{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.95,0.05],[0.05,0.95]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.1],[0.9,0.1]],[[0.1,0.9],[0.1,0.9]]],"param_type":"constant"},{"name":"C","value":[[0.0,3.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":2,"param_type":"constant"},{"name":"num_obs","value":2,"param_type":"constant"},{"name":"num_actions","value":2,"param_type":"constant"},{"name":"num_timesteps","value":50,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":50,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
