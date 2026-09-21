"""
GNN Model: NEST D04 Synthetic 2-State LGSSM
Passive linear-Gaussian state-space model (no control input) generating the
summary indices of the D02 MESSAGEix wrapper blanket over T coupling iterations.
- Hidden state x = (decarb_rate_dev, demand_pressure), dimensionless deviations.
- Observation y = (emissions_index, price_index, objective_index, demand_index): relative indices of total
  emissions e, mean commodity price p, objective J and mean demand d against
  fixed reference values (see the fixture JSON, `blanket.references`).
- cap (emissions cap) is a declared exogenous schedule in the fixture, not a
  variable of this model.
Deliverable D04 (AII); pattern for D07 (3–5 states) and D14 (GNN → RxInfer render).
Generated: 2026-09-11T18:29:39.608319
"""

import numpy as np
from typing import Dict, List, Any

class NESTD04Synthetic2StateLGSSMModel:
    """GNN Model: NEST D04 Synthetic 2-State LGSSM"""

    def __init__(self):
        self.model_name = "NEST D04 Synthetic 2-State LGSSM"
        self.version = "1.0"
        self.annotation = "Passive linear-Gaussian state-space model (no control input) generating the\nsummary indices of the D02 MESSAGEix wrapper blanket over T coupling iterations.\n- Hidden state x = (decarb_rate_dev, demand_pressure), dimensionless deviations.\n- Observation y = (emissions_index, price_index, objective_index, demand_index): relative indices of total\n  emissions e, mean commodity price p, objective J and mean demand d against\n  fixed reference values (see the fixture JSON, `blanket.references`).\n- cap (emissions cap) is a declared exogenous schedule in the fixture, not a\n  variable of this model.\nDeliverable D04 (AII); pattern for D07 (3\u20135 states) and D14 (GNN \u2192 RxInfer render)."

        # Variables
        self.variables = {
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "state transition",
            },
            "H": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 2],
                "description": "observation matrix",
            },
            "Q": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "process-noise covariance",
            },
            "R": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "observation-noise covariance",
            },
            "prior_cov": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "prior covariance over x_1",
            },
            "prior_mean": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2],
                "description": "prior mean over x_1",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "coupling iteration",
            },
            "x": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "hidden state (decarb_rate_dev, demand_pressure)",
            },
            "y": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "blanket summary indices",
            },
        }

        # Parameters
        self.parameters = {
            "F": [[0.85, -0.1], [0.05, 0.9]],
            "H": [[-0.8, 0.5], [0.6, 0.4], [0.3, 0.7], [0.0, 1.0]],
            "Q": [[0.004, 0.0005], [0.0005, 0.006]],
            "R": [[0.0025, 0.0, 0.0, 0.0], [0.0, 0.0036, 0.0, 0.0], [0.0, 0.0, 0.0016, 0.0], [0.0, 0.0, 0.0, 0.0009]],
            "num_commodities": 2,
            "num_observations": 4,
            "num_regions": 1,
            "num_states": 2,
            "num_time_slices": 12,
            "num_timesteps": 24,
            "prior_cov": [[0.02, 0.0], [0.0, 0.02]],
            "prior_mean": [[0.05, 0.0]],
            "random_seed": 20260910,
        }

# MODEL_DATA: {"model_name":"NEST D04 Synthetic 2-State LGSSM","annotation":"Passive linear-Gaussian state-space model (no control input) generating the\nsummary indices of the D02 MESSAGEix wrapper blanket over T coupling iterations.\n- Hidden state x = (decarb_rate_dev, demand_pressure), dimensionless deviations.\n- Observation y = (emissions_index, price_index, objective_index, demand_index): relative indices of total\n  emissions e, mean commodity price p, objective J and mean demand d against\n  fixed reference values (see the fixture JSON, `blanket.references`).\n- cap (emissions cap) is a declared exogenous schedule in the fixture, not a\n  variable of this model.\nDeliverable D04 (AII); pattern for D07 (3\u20135 states) and D14 (GNN \u2192 RxInfer render).","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[4,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[4,4]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[0.85,-0.1],[0.05,0.9]],"param_type":"constant"},{"name":"H","value":[[-0.8,0.5],[0.6,0.4],[0.3,0.7],[0.0,1.0]],"param_type":"constant"},{"name":"Q","value":[[0.004,0.0005],[0.0005,0.006]],"param_type":"constant"},{"name":"R","value":[[0.0025,0.0,0.0,0.0],[0.0,0.0036,0.0,0.0],[0.0,0.0,0.0016,0.0],[0.0,0.0,0.0,0.0009]],"param_type":"constant"},{"name":"prior_mean","value":[[0.05,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[0.02,0.0],[0.0,0.02]],"param_type":"constant"},{"name":"num_timesteps","value":24,"param_type":"constant"},{"name":"random_seed","value":20260910,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":4,"param_type":"constant"},{"name":"num_regions","value":1,"param_type":"constant"},{"name":"num_commodities","value":2,"param_type":"constant"},{"name":"num_time_slices","value":12,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":24,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
