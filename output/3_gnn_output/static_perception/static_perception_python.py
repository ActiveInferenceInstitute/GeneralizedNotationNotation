"""
GNN Model: Static Perception Model
The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- Minimal 2-action transition component B so the model is a complete POMDP
  (renderable and executable by pymdp and the general simulation frameworks)
- Suitable as a minimal baseline and for testing perception-only inference
Generated: 2026-09-05T20:30:45.903753
"""

import numpy as np
from typing import Dict, List, Any

class StaticPerceptionModelModel:
    """GNN Model: Static Perception Model"""

    def __init__(self):
        self.model_name = "Static Perception Model"
        self.version = "1.0"
        self.annotation = "The simplest Active Inference model demonstrating pure perception:\n\n- 2 hidden states mapped to 2 observations via a recognition matrix A\n- Prior D encodes initial beliefs over hidden states\n- Minimal 2-action transition component B so the model is a complete POMDP\n  (renderable and executable by pymdp and the general simulation frameworks)\n- Suitable as a minimal baseline and for testing perception-only inference"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "Recognition/likelihood matrix: P(observation | hidden state)",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [2, 2, 2],
                "description": "Transition matrix: B[next_state, previous_state, actions]",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [2],
                "description": "Preference vector over observations",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Prior belief over hidden states",
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [2, 1],
                "description": "Observation (one-hot encoded)",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Hidden state (posterior belief)",
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Action taken",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.1], [0.2, 0.8]],
            "B": [[[0.95, 0.05], [0.05, 0.95]], [[0.05, 0.95], [0.95, 0.05]]],
            "C": [[0.0, 0.0]],
            "D": [[0.5, 0.5]],
            "num_actions": 2,
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_timesteps": 5,
        }

# MODEL_DATA: {"model_name":"Static Perception Model","annotation":"The simplest Active Inference model demonstrating pure perception:\n\n- 2 hidden states mapped to 2 observations via a recognition matrix A\n- Prior D encodes initial beliefs over hidden states\n- Minimal 2-action transition component B so the model is a complete POMDP\n  (renderable and executable by pymdp and the general simulation frameworks)\n- Suitable as a minimal baseline and for testing perception-only inference","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"annotation":null,"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["u"],"target_variables":["s"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[[0.95,0.05],[0.05,0.95]],[[0.05,0.95],[0.95,0.05]]],"param_type":"constant"},{"name":"C","value":[[0.0,0.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":2,"param_type":"constant"},{"name":"num_obs","value":2,"param_type":"constant"},{"name":"num_actions","value":2,"param_type":"constant"},{"name":"num_timesteps","value":5,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Static","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"PreferenceVector","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"u","ontology_term":"Action","description":null}]}
