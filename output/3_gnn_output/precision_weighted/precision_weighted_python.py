"""
GNN Model: Precision-Weighted Active Inference Agent
An Active Inference agent with explicit precision parameters:
- ω (omega): sensory precision weighting likelihood confidence
- γ (gamma): policy precision controlling action randomness
- β (beta): inverse temperature for policy selection (softmax)
- 3 hidden states, 3 observations, 3 actions (same topology as base POMDP)
- Precision parameters enable modeling of attention and confidence
Generated: 2026-03-05T10:37:35.965803
"""

import numpy as np
from typing import Dict, List, Any

class PrecisionWeightedActiveInferenceAgentModel:
    """GNN Model: Precision-Weighted Active Inference Agent"""

    def __init__(self):
        self.model_name = "Precision-Weighted Active Inference Agent"
        self.version = "1.0"
        self.annotation = "An Active Inference agent with explicit precision parameters:
- ω (omega): sensory precision weighting likelihood confidence
- γ (gamma): policy precision controlling action randomness
- β (beta): inverse temperature for policy selection (softmax)
- 3 hidden states, 3 observations, 3 actions (same topology as base POMDP)
- Precision parameters enable modeling of attention and confidence"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Likelihood matrix (modulated by ω)",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [3, 3, 3],
                "description": "Transition matrix",
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
                "description": "Prior over hidden states",
            },
            "E": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3],
                "description": "Habit (prior over actions)",
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Variational Free Energy",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Expected Free Energy",
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [3, 1],
                "description": "Current observation",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Hidden state distribution",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Next hidden state",
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Selected action",
            },
            "β": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Inverse temperature (β = 1/γ, controls randomness)",
            },
            "γ": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Policy precision (temperature for action selection)",
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3],
                "description": "Policy distribution",
            },
            "ω": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Sensory precision (modulates A matrix confidence)",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
            "B": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]],
            "C": [[0.1, 0.1, 1.0]],
            "D": [[0.333, 0.333, 0.333]],
            "E": [[0.333, 0.333, 0.333]],
            "β": [[0.5]],
            "γ": [[2.0]],
            "ω": [[4.0]],
        }

# MODEL_DATA: {"model_name":"Precision-Weighted Active Inference Agent","annotation":"An Active Inference agent with explicit precision parameters:\n- \u03c9 (omega): sensory precision weighting likelihood confidence\n- \u03b3 (gamma): policy precision controlling action randomness\n- \u03b2 (beta): inverse temperature for policy selection (softmax)\n- 3 hidden states, 3 observations, 3 actions (same topology as base POMDP)\n- Precision parameters enable modeling of attention and confidence","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,3]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"\u03c9","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"\u03b3","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"\u03b2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["\u03c9"],"target_variables":["A"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03b3"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03b2"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03c9"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],"param_type":"constant"},{"name":"B","value":[[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]],[[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C","value":[[0.1,0.1,1.0]],"param_type":"constant"},{"name":"D","value":[[0.333,0.333,0.333]],"param_type":"constant"},{"name":"E","value":[[0.333,0.333,0.333]],"param_type":"constant"},{"name":"\u03c9","value":[[4.0]],"param_type":"constant"},{"name":"\u03b3","value":[[2.0]],"param_type":"constant"},{"name":"\u03b2","value":[[0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"Habit","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"\u03c9","ontology_term":"SensoryPrecision","description":null},{"variable_name":"\u03b3","ontology_term":"PolicyPrecision","description":null},{"variable_name":"\u03b2","ontology_term":"InverseTemperature","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
