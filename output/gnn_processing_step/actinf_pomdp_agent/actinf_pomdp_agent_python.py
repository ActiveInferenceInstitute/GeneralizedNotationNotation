"""
GNN Model: Classic Active Inference POMDP Agent v1

Generated: 2025-07-28T07:42:28.760345
"""

import numpy as np
from typing import Dict, List, Any

class ClassicActiveInferencePOMDPAgentv1Model:
    """GNN Model: Classic Active Inference POMDP Agent v1"""

    def __init__(self):
        self.model_name = "Classic Active Inference POMDP Agent v1"
        self.version = "1.0"
        self.annotation = ""

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "likelihood_matrix",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [3, 3, 3],
                "description": "transition_matrix",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "preference_vector",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [3],
                "description": "prior_vector",
            },
            "E": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3],
                "description": "policy",
            },
        }

        # Parameters
        self.parameters = {
            "A": [],
            "B": [],
            "C": [],
            "D": [],
            "E": [],
        }

# MODEL_DATA: {"model_name":"Classic Active Inference POMDP Agent v1","annotation":"","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,3]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[3]}],"connections":[{"source_variables":["A"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["o"],"connection_type":"directed"},{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["o"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[],"param_type":"constant"},{"name":"B","value":[],"param_type":"constant"},{"name":"C","value":[],"param_type":"constant"},{"name":"D","value":[],"param_type":"constant"},{"name":"E","value":[],"param_type":"constant"}],"equations":[],"time_specification":null,"ontology_mappings":[]}
