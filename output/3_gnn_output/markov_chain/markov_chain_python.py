"""
GNN Model: Simple Markov Chain
This model describes a minimal discrete-time Markov Chain:

- 3 states representing weather (sunny, cloudy, rainy).
- No actions — the system evolves passively.
- Observations = states directly (identity mapping for monitoring).
- Stationary transition matrix with realistic weather dynamics.
- Tests the simplest model structure: passive state evolution with no control.
Generated: 2026-03-06T15:00:16.275359
"""

import numpy as np
from typing import Dict, List, Any

class SimpleMarkovChainModel:
    """GNN Model: Simple Markov Chain"""

    def __init__(self):
        self.model_name = "Simple Markov Chain"
        self.version = "1.0"
        self.annotation = "This model describes a minimal discrete-time Markov Chain:

- 3 states representing weather (sunny, cloudy, rainy).
- No actions — the system evolves passively.
- Observations = states directly (identity mapping for monitoring).
- Stationary transition matrix with realistic weather dynamics.
- Tests the simplest model structure: passive state evolution with no control."

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Observation model (identity for direct monitoring)",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Markov transition matrix",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "Prior over initial states",
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
                "description": "Current state distribution",
            },
            "s_prime": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Next state distribution",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time step",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "B": [[0.7, 0.3, 0.1], [0.2, 0.4, 0.3], [0.1, 0.3, 0.6]],
            "D": [[0.5, 0.3, 0.2]],
        }

# MODEL_DATA: {"model_name":"Simple Markov Chain","annotation":"This model describes a minimal discrete-time Markov Chain:\n\n- 3 states representing weather (sunny, cloudy, rainy).\n- No actions \u2014 the system evolves passively.\n- Observations = states directly (identity mapping for monitoring).\n- Stationary transition matrix with realistic weather dynamics.\n- Tests the simplest model structure: passive state evolution with no control.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"B","value":[[0.7,0.3,0.1],[0.2,0.4,0.3],[0.1,0.3,0.6]],"param_type":"constant"},{"name":"D","value":[[0.5,0.3,0.2]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"EmissionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"InitialStateDistribution","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
