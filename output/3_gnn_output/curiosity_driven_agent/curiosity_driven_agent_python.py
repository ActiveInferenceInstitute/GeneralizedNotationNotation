"""
GNN Model: Curiosity-Driven Active Inference Agent
An Active Inference agent with:
- Explicit epistemic value (information gain / Bayesian surprise) component in G
- Separate instrumental value (preference satisfaction) component
- Precision parameter γ weighting epistemic vs instrumental contributions
- 5 hidden states, 5 observations, 4 actions in a navigation context
- Agent is rewarded for reducing posterior uncertainty
Generated: 2026-03-13T14:15:04.690188
"""

import numpy as np
from typing import Dict, List, Any

class CuriosityDrivenActiveInferenceAgentModel:
    """GNN Model: Curiosity-Driven Active Inference Agent"""

    def __init__(self):
        self.model_name = "Curiosity-Driven Active Inference Agent"
        self.version = "1.0"
        self.annotation = "An Active Inference agent with:
- Explicit epistemic value (information gain / Bayesian surprise) component in G
- Separate instrumental value (preference satisfaction) component
- Precision parameter γ weighting epistemic vs instrumental contributions
- 5 hidden states, 5 observations, 4 actions in a navigation context
- Agent is rewarded for reducing posterior uncertainty"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [5, 5],
                "description": "Likelihood matrix",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [5, 5, 4],
                "description": "Transition matrix (4 actions: up/down/left/right)",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [5],
                "description": "Instrumental preference vector (goal observations)",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [5],
                "description": "Prior over hidden states (uniform = no preference)",
            },
            "E": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4],
                "description": "Habit vector over actions",
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Variational Free Energy for state inference",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Total Expected Free Energy (epistemic + instrumental)",
            },
            "G_epi": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Epistemic value component (information gain)",
            },
            "G_ins": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Instrumental value component (preference satisfaction)",
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [5, 1],
                "description": "Current observation",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [5, 1],
                "description": "Hidden state belief",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [5, 1],
                "description": "Next hidden state belief",
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
                "description": "Selected action",
            },
            "γ": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Precision weighting epistemic vs instrumental value",
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4],
                "description": "Policy distribution over actions",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.025, 0.025, 0.025, 0.025], [0.025, 0.9, 0.025, 0.025, 0.025], [0.025, 0.025, 0.9, 0.025, 0.025], [0.025, 0.025, 0.025, 0.9, 0.025], [0.025, 0.025, 0.025, 0.025, 0.9]],
            "B": [[[0.9, 0.1, 0.0, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0, 0.0], [0.0, 0.1, 0.8, 0.1, 0.0], [0.0, 0.0, 0.1, 0.8, 0.1], [0.0, 0.0, 0.0, 0.1, 0.9]], [[0.9, 0.1, 0.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.0, 0.1, 0.9]], [[0.9, 0.0, 0.0, 0.0, 0.1], [0.0, 0.9, 0.0, 0.0, 0.1], [0.0, 0.0, 0.9, 0.0, 0.1], [0.0, 0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.0, 0.0, 1.0]]],
            "C": [[-2.0, -2.0, -2.0, -2.0, 2.0]],
            "D": [[0.2, 0.2, 0.2, 0.2, 0.2]],
            "E": [[0.25, 0.25, 0.25, 0.25]],
            "γ": [[1.0]],
        }

# MODEL_DATA: {"model_name":"Curiosity-Driven Active Inference Agent","annotation":"An Active Inference agent with:\n- Explicit epistemic value (information gain / Bayesian surprise) component in G\n- Separate instrumental value (preference satisfaction) component\n- Precision parameter \u03b3 weighting epistemic vs instrumental contributions\n- 5 hidden states, 5 observations, 4 actions in a navigation context\n- Agent is rewarded for reducing posterior uncertainty","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[5,5]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[5,5,4]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[5]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[5]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[5,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[5,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[5,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"G_epi","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"G_ins","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"\u03b3","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G_ins"],"connection_type":"directed"},{"source_variables":["G_epi"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G_ins"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["\u03b3"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.025,0.025,0.025,0.025],[0.025,0.9,0.025,0.025,0.025],[0.025,0.025,0.9,0.025,0.025],[0.025,0.025,0.025,0.9,0.025],[0.025,0.025,0.025,0.025,0.9]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.1,0.0,0.0,0.0],[0.1,0.8,0.1,0.0,0.0],[0.0,0.1,0.8,0.1,0.0],[0.0,0.0,0.1,0.8,0.1],[0.0,0.0,0.0,0.1,0.9]],[[0.9,0.1,0.0,0.0,0.0],[0.0,0.9,0.1,0.0,0.0],[0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.9,0.1],[0.0,0.0,0.0,0.0,1.0]],[[1.0,0.0,0.0,0.0,0.0],[0.1,0.9,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0],[0.0,0.0,0.1,0.9,0.0],[0.0,0.0,0.0,0.1,0.9]],[[0.9,0.0,0.0,0.0,0.1],[0.0,0.9,0.0,0.0,0.1],[0.0,0.0,0.9,0.0,0.1],[0.0,0.0,0.0,0.9,0.1],[0.0,0.0,0.0,0.0,1.0]]],"param_type":"constant"},{"name":"C","value":[[-2.0,-2.0,-2.0,-2.0,2.0]],"param_type":"constant"},{"name":"D","value":[[0.2,0.2,0.2,0.2,0.2]],"param_type":"constant"},{"name":"E","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"\u03b3","value":[[1.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":30,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"Habit","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"G_epi","ontology_term":"EpistemicValue","description":null},{"variable_name":"G_ins","ontology_term":"InstrumentalValue","description":null},{"variable_name":"\u03b3","ontology_term":"PrecisionParameter","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
