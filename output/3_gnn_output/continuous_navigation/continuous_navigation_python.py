"""
GNN Model: Continuous State Navigation Agent
A continuous-state Active Inference navigation agent rendered as a native
linear-Gaussian state-space model (LGSSM):
- Hidden state x = (x, y): the continuous 2D position of the navigator.
- Observation y: noisy readings of the 2D position (identity readout).
- Control input u: a goal-seeking command added to the state each step.
- The controller closes the loop on beliefs — it pushes the filtered posterior
  mean toward the preferred position goal_mean = (2.0, 2.0) with proportional
  gain control_gain = 0.3, i.e. u_t = control_gain * (goal_mean - mu_t).
Generated: 2026-09-05T20:30:55.007131
"""

import numpy as np
from typing import Dict, List, Any

class ContinuousStateNavigationAgentModel:
    """GNN Model: Continuous State Navigation Agent"""

    def __init__(self):
        self.model_name = "Continuous State Navigation Agent"
        self.version = "1.0"
        self.annotation = "A continuous-state Active Inference navigation agent rendered as a native\nlinear-Gaussian state-space model (LGSSM):\n- Hidden state x = (x, y): the continuous 2D position of the navigator.\n- Observation y: noisy readings of the 2D position (identity readout).\n- Control input u: a goal-seeking command added to the state each step.\n- The controller closes the loop on beliefs \u2014 it pushes the filtered posterior\n  mean toward the preferred position goal_mean = (2.0, 2.0) with proportional\n  gain control_gain = 0.3, i.e. u_t = control_gain * (goal_mean - mu_t)."

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
            "control_gain": {
                "type": "action",
                "data_type": "float",
                "dimensions": [1],
                "description": "scalar proportional gain",
            },
            "goal_mean": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "preferred state (goal)",
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
            "u": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "control input",
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
            "F": [[1.0, 0.0], [0.0, 1.0]],
            "H": [[1.0, 0.0], [0.0, 1.0]],
            "Q": [[0.05, 0.0], [0.0, 0.05]],
            "R": [[0.1, 0.0], [0.0, 0.1]],
            "control_gain": [[0.3]],
            "dt": 0.1,
            "goal_mean": [[2.0, 2.0]],
            "num_observations": 2,
            "num_states": 2,
            "num_timesteps": 15,
            "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
            "prior_mean": [[0.0, 0.0]],
            "random_seed": 42,
        }

# MODEL_DATA: {"model_name":"Continuous State Navigation Agent","annotation":"A continuous-state Active Inference navigation agent rendered as a native\nlinear-Gaussian state-space model (LGSSM):\n- Hidden state x = (x, y): the continuous 2D position of the navigator.\n- Observation y: noisy readings of the 2D position (identity readout).\n- Control input u: a goal-seeking command added to the state each step.\n- The controller closes the loop on beliefs \u2014 it pushes the filtered posterior\n  mean toward the preferred position goal_mean = (2.0, 2.0) with proportional\n  gain control_gain = 0.3, i.e. u_t = control_gain * (goal_mean - mu_t).","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"goal_mean","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"control_gain","var_type":"action","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["u"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["goal_mean"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["control_gain"],"target_variables":["u"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"H","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"Q","value":[[0.05,0.0],[0.0,0.05]],"param_type":"constant"},{"name":"R","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"prior_mean","value":[[0.0,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[0.5,0.0],[0.0,0.5]],"param_type":"constant"},{"name":"goal_mean","value":[[2.0,2.0]],"param_type":"constant"},{"name":"control_gain","value":[[0.3]],"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"},{"name":"dt","value":0.1,"param_type":"constant"},{"name":"random_seed","value":42,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":2,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"goal_mean","ontology_term":"PreferredState","description":null},{"variable_name":"control_gain","ontology_term":"ControlGain","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"u","ontology_term":"ControlInput","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
