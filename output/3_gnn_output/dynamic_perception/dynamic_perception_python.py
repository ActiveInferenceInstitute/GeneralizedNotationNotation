"""
GNN Model: Dynamic Perception Model
A dynamic perception model extending the static model with temporal dynamics:

- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via recognition matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations
Generated: 2026-03-24T13:57:11.263344
"""

import numpy as np
from typing import Dict, List, Any

class DynamicPerceptionModelModel:
    """GNN Model: Dynamic Perception Model"""

    def __init__(self):
        self.model_name = "Dynamic Perception Model"
        self.version = "1.0"
        self.annotation = "A dynamic perception model extending the static model with temporal dynamics:

- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via recognition matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "Recognition matrix: P(observation | hidden state)",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "Transition matrix: P(s_{t+1} | s_t) — no action dependence",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Prior over initial hidden states",
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Variational Free Energy (negative ELBO)",
            },
            "o_t": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [2, 1],
                "description": "Observation at time t",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Hidden state belief at time t+1",
            },
            "s_t": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Hidden state belief at time t",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time index",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.1], [0.2, 0.8]],
            "B": [[0.7, 0.3], [0.3, 0.7]],
            "D": [[0.5, 0.5]],
        }

# MODEL_DATA: {"model_name":"Dynamic Perception Model","annotation":"A dynamic perception model extending the static model with temporal dynamics:\n\n- 2 hidden states evolving over discrete time via transition matrix B\n- 2 observations generated from states via recognition matrix A\n- Prior D constrains the initial hidden state\n- No action selection \u2014 the agent passively observes a changing world\n- Demonstrates belief updating (state inference) across time steps\n- Suitable for tracking hidden sources from noisy observations","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"s_t","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_t","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s_t"],"connection_type":"directed"},{"source_variables":["s_t"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o_t"],"connection_type":"undirected"},{"source_variables":["s_t"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s_t"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_t"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[0.7,0.3],[0.3,0.7]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":10,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s_t","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o_t","ontology_term":"Observation","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
