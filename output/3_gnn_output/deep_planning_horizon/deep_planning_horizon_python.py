"""
GNN Model: Deep Planning Horizon POMDP
An Active Inference POMDP with deep (T=5) planning horizon:
- Evaluates policies over 5 future timesteps before acting
- Uses rollout Expected Free Energy accumulation
- 4 hidden states, 4 observations, 4 actions
- Each action policy is a sequence of T actions: π = [a_1, a_2, ..., a_T]
- Enables sophisticated multi-step reasoning and delayed reward attribution
Generated: 2026-03-13T18:17:53.730408
"""

import numpy as np
from typing import Dict, List, Any

class DeepPlanningHorizonPOMDPModel:
    """GNN Model: Deep Planning Horizon POMDP"""

    def __init__(self):
        self.model_name = "Deep Planning Horizon POMDP"
        self.version = "1.0"
        self.annotation = "An Active Inference POMDP with deep (T=5) planning horizon:
- Evaluates policies over 5 future timesteps before acting
- Uses rollout Expected Free Energy accumulation
- 4 hidden states, 4 observations, 4 actions
- Each action policy is a sequence of T actions: π = [a_1, a_2, ..., a_T]
- Enables sophisticated multi-step reasoning and delayed reward attribution"

        # Variables
        self.variables = {
            "A": {
                "type": "likelihood_matrix",
                "data_type": "float",
                "dimensions": [4, 4],
                "description": "Likelihood matrix",
            },
            "B": {
                "type": "transition_matrix",
                "data_type": "float",
                "dimensions": [4, 4, 4],
                "description": "Transition matrix (4 actions)",
            },
            "C": {
                "type": "preference_vector",
                "data_type": "float",
                "dimensions": [4],
                "description": "Preferences (per observation)",
            },
            "D": {
                "type": "prior_vector",
                "data_type": "float",
                "dimensions": [4],
                "description": "Prior over initial states",
            },
            "E": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [64],
                "description": "Habit prior over policies (4^3 = 64 short policies)",
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Variational Free Energy for current state",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [64],
                "description": "Cumulative EFE (sum over horizon)",
            },
            "G_tau1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [64],
                "description": "EFE contribution at tau=1",
            },
            "G_tau2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [64],
                "description": "EFE contribution at tau=2",
            },
            "G_tau3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [64],
                "description": "EFE contribution at tau=3",
            },
            "G_tau4": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [64],
                "description": "EFE contribution at tau=4",
            },
            "G_tau5": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [64],
                "description": "EFE contribution at tau=5",
            },
            "o": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [4, 1],
                "description": "Current observation",
            },
            "s": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Current hidden state belief",
            },
            "s_tau1": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Predicted state at tau=1",
            },
            "s_tau2": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Predicted state at tau=2",
            },
            "s_tau3": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Predicted state at tau=3",
            },
            "s_tau4": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Predicted state at tau=4",
            },
            "s_tau5": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Predicted state at tau=5",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Discrete time step (action timestep)",
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Selected first action from best policy",
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [64],
                "description": "Policy distribution (over T-step action sequences)",
            },
        }

        # Parameters
        self.parameters = {
            "A": [[0.9, 0.05, 0.025, 0.025], [0.05, 0.9, 0.025, 0.025], [0.025, 0.025, 0.9, 0.05], [0.025, 0.025, 0.05, 0.9]],
            "B": [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.9, 0.1], [0.1, 0.0, 0.0, 0.9]], [[0.9, 0.0, 0.0, 0.1], [0.1, 0.9, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0], [0.0, 0.0, 0.1, 0.9]], [[0.8, 0.1, 0.1, 0.0], [0.0, 0.8, 0.1, 0.1], [0.1, 0.0, 0.8, 0.1], [0.1, 0.1, 0.0, 0.8]], [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]],
            "C": [[-1.0, -0.5, -0.5, 2.0]],
            "D": [[0.25, 0.25, 0.25, 0.25]],
        }

# MODEL_DATA: {"model_name":"Deep Planning Horizon POMDP","annotation":"An Active Inference POMDP with deep (T=5) planning horizon:\n- Evaluates policies over 5 future timesteps before acting\n- Uses rollout Expected Free Energy accumulation\n- 4 hidden states, 4 observations, 4 actions\n- Each action policy is a sequence of T actions: \u03c0 = [a_1, a_2, ..., a_T]\n- Enables sophisticated multi-step reasoning and delayed reward attribution","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[4,4]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[4,4,4]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[4]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[4]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[64]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[64]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"s_tau1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_tau2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_tau3","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_tau4","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_tau5","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"G_tau1","var_type":"hidden_state","data_type":"float","dimensions":[64]},{"name":"G_tau2","var_type":"hidden_state","data_type":"float","dimensions":[64]},{"name":"G_tau3","var_type":"hidden_state","data_type":"float","dimensions":[64]},{"name":"G_tau4","var_type":"hidden_state","data_type":"float","dimensions":[64]},{"name":"G_tau5","var_type":"hidden_state","data_type":"float","dimensions":[64]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[64]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["s_tau1"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_tau1"],"connection_type":"directed"},{"source_variables":["s_tau1"],"target_variables":["s_tau2"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_tau2"],"connection_type":"directed"},{"source_variables":["s_tau2"],"target_variables":["s_tau3"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_tau3"],"connection_type":"directed"},{"source_variables":["s_tau3"],"target_variables":["s_tau4"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_tau4"],"connection_type":"directed"},{"source_variables":["s_tau4"],"target_variables":["s_tau5"],"connection_type":"directed"},{"source_variables":["A"],"target_variables":["s_tau1"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["s_tau2"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["s_tau3"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["s_tau4"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["s_tau5"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G_tau1"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G_tau2"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G_tau3"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G_tau4"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G_tau5"],"connection_type":"directed"},{"source_variables":["G_tau1"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G_tau2"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G_tau3"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G_tau4"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G_tau5"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.05,0.025,0.025],[0.05,0.9,0.025,0.025],[0.025,0.025,0.9,0.05],[0.025,0.025,0.05,0.9]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.0,0.8,0.1,0.1],[0.1,0.0,0.8,0.1],[0.1,0.1,0.0,0.8]],[[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.1,0.1,0.7,0.1],[0.1,0.1,0.1,0.7]]],"param_type":"constant"},{"name":"C","value":[[-1.0,-0.5,-0.5,2.0]],"param_type":"constant"},{"name":"D","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"PolicyPrior","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicySequenceDistribution","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"G","ontology_term":"CumulativeExpectedFreeEnergy","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
