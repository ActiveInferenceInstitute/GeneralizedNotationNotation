"""
GNN Model: Stochastic Continuous Dynamics Agent
A continuous-state Active Inference agent whose dynamics carry explicit process
and observation noise, rendered as a native linear-Gaussian state-space model
(LGSSM). The agent runs passively — it has no control input:
- Hidden state x = (position, velocity): the Euler-discretized (dt = 0.1) SDE.
- Observation y: two noisy readouts, both reading the position.
- Q is the process-noise covariance (inverse process precision); R is the
  observation-noise covariance (inverse observation precision).
Generated: 2026-09-06T11:50:16.039178
"""

import numpy as np
from typing import Dict, List, Any

class StochasticContinuousDynamicsAgentModel:
    """GNN Model: Stochastic Continuous Dynamics Agent"""

    def __init__(self):
        self.model_name = "Stochastic Continuous Dynamics Agent"
        self.version = "1.0"
        self.annotation = "A continuous-state Active Inference agent whose dynamics carry explicit process\nand observation noise, rendered as a native linear-Gaussian state-space model\n(LGSSM). The agent runs passively \u2014 it has no control input:\n- Hidden state x = (position, velocity): the Euler-discretized (dt = 0.1) SDE.\n- Observation y: two noisy readouts, both reading the position.\n- Q is the process-noise covariance (inverse process precision); R is the\n  observation-noise covariance (inverse observation precision)."

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
                "dimensions": [2, 2],
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
                "dimensions": [2, 2],
                "description": "observation-noise covariance",
            },
            "prior_cov": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2, 2],
                "description": "prior covariance over the initial latent state",
            },
            "prior_mean": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [2],
                "description": "prior mean over the initial latent state",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "discrete time step",
            },
            "x": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "continuous latent state",
            },
            "y": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "continuous observation",
            },
        }

        # Parameters
        self.parameters = {
            "F": [[1.0, 0.1], [0.0, 0.9]],
            "H": [[1.0, 0.0], [1.0, 0.0]],
            "Q": [[0.1, 0.0], [0.0, 0.1]],
            "R": [[0.2, 0.0], [0.0, 0.2]],
            "dt": 0.1,
            "num_observations": 2,
            "num_states": 2,
            "num_timesteps": 15,
            "prior_cov": [[0.5, 0.0], [0.0, 1.0]],
            "prior_mean": [[0.0, 0.0]],
            "random_seed": 42,
        }

# MODEL_DATA: {"model_name":"Stochastic Continuous Dynamics Agent","annotation":"A continuous-state Active Inference agent whose dynamics carry explicit process\nand observation noise, rendered as a native linear-Gaussian state-space model\n(LGSSM). The agent runs passively \u2014 it has no control input:\n- Hidden state x = (position, velocity): the Euler-discretized (dt = 0.1) SDE.\n- Observation y: two noisy readouts, both reading the position.\n- Q is the process-noise covariance (inverse process precision); R is the\n  observation-noise covariance (inverse observation precision).","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[1.0,0.1],[0.0,0.9]],"param_type":"constant"},{"name":"H","value":[[1.0,0.0],[1.0,0.0]],"param_type":"constant"},{"name":"Q","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"R","value":[[0.2,0.0],[0.0,0.2]],"param_type":"constant"},{"name":"prior_mean","value":[[0.0,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[0.5,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"},{"name":"dt","value":0.1,"param_type":"constant"},{"name":"random_seed","value":42,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":2,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
