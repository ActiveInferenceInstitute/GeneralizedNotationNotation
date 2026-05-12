"""
GNN Model: Factorized Posterior Agent
A mean-field factorized POMDP agent. The joint posterior over two
independent state factors `s_1` (location) and `s_2` (goal identity) is
approximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).
This is the canonical simplification used in variational inference when
exact joint posteriors are computationally intractable.

- Two state factors: location (4 states), goal (2 states)
- Two observation modalities: visual (3 obs), proprioceptive (2 obs)
- Separate transition matrices B_1 (location × action) and B_2 (goal is static)
- Explicit factorization declared in ## Equations
- Tests multi-factor / multi-modality handling in the parser
Generated: 2026-05-12T07:28:01.836205
"""

import numpy as np
from typing import Dict, List, Any

class FactorizedPosteriorAgentModel:
    """GNN Model: Factorized Posterior Agent"""

    def __init__(self):
        self.model_name = "Factorized Posterior Agent"
        self.version = "1.0"
        self.annotation = "A mean-field factorized POMDP agent. The joint posterior over two
independent state factors `s_1` (location) and `s_2` (goal identity) is
approximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).
This is the canonical simplification used in variational inference when
exact joint posteriors are computationally intractable.

- Two state factors: location (4 states), goal (2 states)
- Two observation modalities: visual (3 obs), proprioceptive (2 obs)
- Separate transition matrices B_1 (location × action) and B_2 (goal is static)
- Explicit factorization declared in ## Equations
- Tests multi-factor / multi-modality handling in the parser"

        # Variables
        self.variables = {
            "A_m0": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3, 4, 2],
                "description": "Visual likelihood: depends on both factors",
            },
            "A_m1": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2, 4],
                "description": "Proprioceptive likelihood: depends only on location",
            },
            "B_f0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4, 3],
                "description": "Location transitions (depends on action)",
            },
            "B_f1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "Goal transitions (typically identity — goal is static)",
            },
            "C_m0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Visual preferences",
            },
            "C_m1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Proprioceptive preferences",
            },
            "D_f0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Prior over locations (uniform)",
            },
            "D_f1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Prior over goals",
            },
            "o_m0": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [3, 1],
                "description": "Modality 0: visual observation (3 visual cues)",
            },
            "o_m1": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [2, 1],
                "description": "Modality 1: proprioceptive observation (2 body states)",
            },
            "s_f0": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Factor 0: agent location (4 possible positions)",
            },
            "s_f1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Factor 1: goal identity (2 possible goals)",
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [3, 1],
                "description": "3 possible actions: stay, forward, backward",
            },
        }

        # Parameters
        self.parameters = {
            "A_m0": [[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.2, 0.2, 0.8, 0.8]], [[0.1, 0.7, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.2, 0.2, 0.8, 0.8]]],
            "A_m1": [[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]],
            "C_m0": [[0.0, 0.0, 1.0]],
            "C_m1": [[0.5, 0.5]],
            "D_f0": [[0.25, 0.25, 0.25, 0.25]],
            "D_f1": [[0.6, 0.4]],
            "num_actions": 3,
            "num_factors": 2,
            "num_hidden_states_factor0": 4,
            "num_hidden_states_factor1": 2,
            "num_modalities": 2,
            "num_obs_modality0": 3,
            "num_obs_modality1": 2,
            "num_timesteps": 15,
        }

# MODEL_DATA: {"model_name":"Factorized Posterior Agent","annotation":"A mean-field factorized POMDP agent. The joint posterior over two\nindependent state factors `s_1` (location) and `s_2` (goal identity) is\napproximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).\nThis is the canonical simplification used in variational inference when\nexact joint posteriors are computationally intractable.\n\n- Two state factors: location (4 states), goal (2 states)\n- Two observation modalities: visual (3 obs), proprioceptive (2 obs)\n- Separate transition matrices B_1 (location \u00d7 action) and B_2 (goal is static)\n- Explicit factorization declared in ## Equations\n- Tests multi-factor / multi-modality handling in the parser","variables":[{"name":"s_f0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_f1","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_m0","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"o_m1","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[3,1]},{"name":"A_m0","var_type":"action","data_type":"float","dimensions":[3,4,2]},{"name":"A_m1","var_type":"action","data_type":"float","dimensions":[2,4]},{"name":"B_f0","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"B_f1","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"D_f0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"D_f1","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"C_m0","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"C_m1","var_type":"hidden_state","data_type":"float","dimensions":[2,1]}],"connections":[{"source_variables":["D_f0"],"target_variables":["s_f0"],"connection_type":"directed"},{"source_variables":["D_f1"],"target_variables":["s_f1"],"connection_type":"directed"},{"source_variables":["s_f0","u"],"target_variables":["B_f0"],"connection_type":"directed"},{"source_variables":["B_f0"],"target_variables":["s_f0"],"connection_type":"directed"},{"source_variables":["s_f1"],"target_variables":["B_f1"],"connection_type":"directed"},{"source_variables":["B_f1"],"target_variables":["s_f1"],"connection_type":"directed"},{"source_variables":["s_f0","s_f1"],"target_variables":["A_m0"],"connection_type":"directed"},{"source_variables":["A_m0"],"target_variables":["o_m0"],"connection_type":"directed"},{"source_variables":["s_f0"],"target_variables":["A_m1"],"connection_type":"directed"},{"source_variables":["A_m1"],"target_variables":["o_m1"],"connection_type":"directed"},{"source_variables":["C_m0"],"target_variables":["o_m0"],"connection_type":"undirected"},{"source_variables":["C_m1"],"target_variables":["o_m1"],"connection_type":"undirected"}],"parameters":[{"name":"A_m0","value":[[[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.2,0.2,0.8,0.8]],[[0.1,0.7,0.1,0.1],[0.7,0.1,0.1,0.1],[0.2,0.2,0.8,0.8]]],"param_type":"constant"},{"name":"A_m1","value":[[0.9,0.1,0.1,0.1],[0.1,0.9,0.9,0.9]],"param_type":"constant"},{"name":"D_f0","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"D_f1","value":[[0.6,0.4]],"param_type":"constant"},{"name":"C_m0","value":[[0.0,0.0,1.0]],"param_type":"constant"},{"name":"C_m1","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states_factor0","value":4,"param_type":"constant"},{"name":"num_hidden_states_factor1","value":2,"param_type":"constant"},{"name":"num_obs_modality0","value":3,"param_type":"constant"},{"name":"num_obs_modality1","value":2,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_factors","value":2,"param_type":"constant"},{"name":"num_modalities","value":2,"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":"DiscreteTime","horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"s_f0","ontology_term":"HiddenStateFactor0","description":null},{"variable_name":"s_f1","ontology_term":"HiddenStateFactor1","description":null},{"variable_name":"o_m0","ontology_term":"ObservationModality0","description":null},{"variable_name":"o_m1","ontology_term":"ObservationModality1","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"A_m0","ontology_term":"LikelihoodMatrixModality0","description":null},{"variable_name":"A_m1","ontology_term":"LikelihoodMatrixModality1","description":null},{"variable_name":"B_f0","ontology_term":"TransitionMatrixFactor0","description":null},{"variable_name":"B_f1","ontology_term":"TransitionMatrixFactor1","description":null},{"variable_name":"D_f0","ontology_term":"PriorFactor0","description":null},{"variable_name":"D_f1","ontology_term":"PriorFactor1","description":null},{"variable_name":"C_m0","ontology_term":"PreferenceModality0","description":null},{"variable_name":"C_m1","ontology_term":"PreferenceModality1","description":null}]}
