"""
GNN Model: Predictive Coding Active Inference Agent
A continuous-state Active Inference agent implementing predictive coding:

- Two-level predictive hierarchy: sensory predictions and dynamics predictions
- Prediction errors drive belief updating via gradient descent on free energy
- Precision-weighted prediction errors enable attentional modulation
- Sensory level: predicts observations from hidden causes
- Dynamics level: predicts state evolution from generative dynamics
- Action minimizes expected free energy by changing sensory input
- Uses generalized coordinates of motion (position, velocity, acceleration)
- Demonstrates the core predictive processing framework underlying Active Inference
Generated: 2026-03-05T10:37:36.358891
"""

import numpy as np
from typing import Dict, List, Any

class PredictiveCodingActiveInferenceAgentModel:
    """GNN Model: Predictive Coding Active Inference Agent"""

    def __init__(self):
        self.model_name = "Predictive Coding Active Inference Agent"
        self.version = "1.0"
        self.annotation = "A continuous-state Active Inference agent implementing predictive coding:

- Two-level predictive hierarchy: sensory predictions and dynamics predictions
- Prediction errors drive belief updating via gradient descent on free energy
- Precision-weighted prediction errors enable attentional modulation
- Sensory level: predicts observations from hidden causes
- Dynamics level: predicts state evolution from generative dynamics
- Action minimizes expected free energy by changing sensory input
- Uses generalized coordinates of motion (position, velocity, acceleration)
- Demonstrates the core predictive processing framework underlying Active Inference"

        # Variables
        self.variables = {
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Variational Free Energy (scalar)",
            },
            "F_d": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Dynamics contribution to VFE",
            },
            "F_s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Sensory contribution to VFE",
            },
            "Pi_d": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Dynamics precision: confidence in dynamics model",
            },
            "Pi_s": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Sensory precision: confidence in observations",
            },
            "Sigma": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 3],
                "description": "Covariance of hidden cause belief",
            },
            "e_d": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Dynamics prediction error: mu_dot - f(mu)",
            },
            "e_s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Sensory prediction error: o - g(mu)",
            },
            "f_params": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [9],
                "description": "Dynamics parameters: mu_dot = f(mu, f_params) + noise",
            },
            "g_params": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [12],
                "description": "Sensory mapping parameters: o = g(mu, g_params) + noise",
            },
            "mu": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "State belief: mean of hidden cause (3D)",
            },
            "mu_dot": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Velocity of hidden cause (first temporal derivative)",
            },
            "mu_star": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Desired state (prior expectation / set-point)",
            },
            "o": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Continuous observation vector (4D sensory input)",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Continuous time",
            },
            "u": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Continuous action (2D motor command)",
            },
        }

        # Parameters
        self.parameters = {
            "Pi_d": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "Pi_s": [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 8.0, 0.0], [0.0, 0.0, 0.0, 8.0]],
            "Sigma": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "mu": [[0.0], [0.0], [0.0]],
            "mu_dot": [[0.0], [0.0], [0.0]],
            "mu_star": [[1.0], [1.0], [0.5]],
        }

# MODEL_DATA: {"model_name":"Predictive Coding Active Inference Agent","annotation":"A continuous-state Active Inference agent implementing predictive coding:\n\n- Two-level predictive hierarchy: sensory predictions and dynamics predictions\n- Prediction errors drive belief updating via gradient descent on free energy\n- Precision-weighted prediction errors enable attentional modulation\n- Sensory level: predicts observations from hidden causes\n- Dynamics level: predicts state evolution from generative dynamics\n- Action minimizes expected free energy by changing sensory input\n- Uses generalized coordinates of motion (position, velocity, acceleration)\n- Demonstrates the core predictive processing framework underlying Active Inference","variables":[{"name":"mu","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"mu_dot","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Sigma","var_type":"hidden_state","data_type":"float","dimensions":[3,3]},{"name":"e_s","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"e_d","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"g_params","var_type":"hidden_state","data_type":"float","dimensions":[12]},{"name":"f_params","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"Pi_s","var_type":"policy","data_type":"float","dimensions":[4,4]},{"name":"Pi_d","var_type":"policy","data_type":"float","dimensions":[3,3]},{"name":"o","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"u","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"F_s","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"F_d","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"mu_star","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"t","var_type":"hidden_state","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["mu"],"target_variables":["g_params"],"connection_type":"undirected"},{"source_variables":["g_params"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["e_s"],"connection_type":"undirected"},{"source_variables":["mu"],"target_variables":["e_s"],"connection_type":"undirected"},{"source_variables":["Pi_s"],"target_variables":["e_s"],"connection_type":"undirected"},{"source_variables":["e_s"],"target_variables":["F_s"],"connection_type":"undirected"},{"source_variables":["mu"],"target_variables":["f_params"],"connection_type":"undirected"},{"source_variables":["f_params"],"target_variables":["mu_dot"],"connection_type":"undirected"},{"source_variables":["mu_dot"],"target_variables":["e_d"],"connection_type":"undirected"},{"source_variables":["Pi_d"],"target_variables":["e_d"],"connection_type":"undirected"},{"source_variables":["e_d"],"target_variables":["F_d"],"connection_type":"undirected"},{"source_variables":["F_s"],"target_variables":["F"],"connection_type":"directed"},{"source_variables":["F_d"],"target_variables":["F"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["mu"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["mu_star"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["mu"],"target_variables":["Sigma"],"connection_type":"undirected"},{"source_variables":["Sigma"],"target_variables":["Pi_s"],"connection_type":"undirected"}],"parameters":[{"name":"mu","value":[[0.0],[0.0],[0.0]],"param_type":"constant"},{"name":"mu_dot","value":[[0.0],[0.0],[0.0]],"param_type":"constant"},{"name":"Sigma","value":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"Pi_s","value":[[8.0,0.0,0.0,0.0],[0.0,8.0,0.0,0.0],[0.0,0.0,8.0,0.0],[0.0,0.0,0.0,8.0]],"param_type":"constant"},{"name":"Pi_d","value":[[4.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,4.0]],"param_type":"constant"},{"name":"mu_star","value":[[1.0],[1.0],[0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"5.0","step_size":null},"ontology_mappings":[{"variable_name":"mu","ontology_term":"BeliefMean","description":null},{"variable_name":"mu_dot","ontology_term":"BeliefVelocity","description":null},{"variable_name":"Sigma","ontology_term":"BeliefCovariance","description":null},{"variable_name":"e_s","ontology_term":"SensoryPredictionError","description":null},{"variable_name":"e_d","ontology_term":"DynamicPredictionError","description":null},{"variable_name":"g_params","ontology_term":"SensoryMappingParameters","description":null},{"variable_name":"f_params","ontology_term":"DynamicsParameters","description":null},{"variable_name":"Pi_s","ontology_term":"SensoryPrecision","description":null},{"variable_name":"Pi_d","ontology_term":"DynamicPrecision","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"u","ontology_term":"ContinuousAction","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"F_s","ontology_term":"SensoryFreeEnergy","description":null},{"variable_name":"F_d","ontology_term":"DynamicFreeEnergy","description":null},{"variable_name":"mu_star","ontology_term":"PriorExpectation","description":null},{"variable_name":"t","ontology_term":"ContinuousTime","description":null}]}
