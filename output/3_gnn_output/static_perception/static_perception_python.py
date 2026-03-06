"""
GNN Model: Static Perception Model
The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- No temporal dynamics — single-shot inference
- Demonstrates the core observation model: P(o|s) = A
- Suitable as a minimal baseline and for testing perception-only inference
Generated: 2026-03-06T09:42:42.646120
"""

import numpy as np
from typing import Dict, List, Any

class StaticPerceptionModelModel:
    """GNN Model: Static Perception Model"""

    def __init__(self):
        self.model_name = "Static Perception Model"
        self.version = "1.0"
        self.annotation = "The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- No temporal dynamics — single-shot inference
- Demonstrates the core observation model: P(o|s) = A
- Suitable as a minimal baseline and for testing perception-only inference"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "Recognition/likelihood matrix: P(observation | hidden state)",
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
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.1], [0.2, 0.8]],
            "D": [[0.5, 0.5]],
        }

# MODEL_DATA: {"model_name":"Static Perception Model","annotation":"The simplest Active Inference model demonstrating pure perception:\n\n- 2 hidden states mapped to 2 observations via a recognition matrix A\n- Prior D encodes initial beliefs over hidden states\n- No temporal dynamics \u2014 single-shot inference\n- Demonstrates the core observation model: P(o|s) = A\n- Suitable as a minimal baseline and for testing perception-only inference","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Static","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null}]}
