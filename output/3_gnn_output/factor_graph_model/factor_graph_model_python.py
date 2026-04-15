"""
GNN Model: Factor Graph Active Inference Model
A factor graph decomposition of an Active Inference generative model with:
- Two independent observation modalities (visual and proprioceptive)
- Two independent hidden state factors (position and velocity)
- Factored joint distribution: P(o,s) = P(o_vis|s_pos) * P(o_prop|s_vel) * P(s_pos|s_vel) * P(s_vel)
- Variable nodes: observation and state variables
- Factor nodes: likelihood and transition factors
- Enables modality-specific processing and efficient belief propagation
Generated: 2026-04-15T12:25:54.523624
"""

import numpy as np
from typing import Dict, List, Any

class FactorGraphActiveInferenceModelModel:
    """GNN Model: Factor Graph Active Inference Model"""

    def __init__(self):
        self.model_name = "Factor Graph Active Inference Model"
        self.version = "1.0"
        self.annotation = "A factor graph decomposition of an Active Inference generative model with:
- Two independent observation modalities (visual and proprioceptive)
- Two independent hidden state factors (position and velocity)
- Factored joint distribution: P(o,s) = P(o_vis|s_pos) * P(o_prop|s_vel) * P(s_pos|s_vel) * P(s_vel)
- Variable nodes: observation and state variables
- Factor nodes: likelihood and transition factors
- Enables modality-specific processing and efficient belief propagation"

        # Variables
        self.variables = {
            "A_prop": {
                "type": "action",
                "data_type": "float",
                "dimensions": [4, 2],
                "description": "Proprioceptive likelihood: P(o_prop | s_vel)",
            },
            "A_vis": {
                "type": "action",
                "data_type": "float",
                "dimensions": [6, 3],
                "description": "Visual likelihood factor: P(o_vis | s_pos)",
            },
            "B_pos": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 3, 2],
                "description": "Position transition factor",
            },
            "B_vel": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 2, 1],
                "description": "Velocity transition (action-independent)",
            },
            "C_prop": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [4],
                "description": "Proprioceptive preferences (comfort)",
            },
            "C_vis": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [6],
                "description": "Visual preferences (goal location)",
            },
            "D_pos": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Prior over position",
            },
            "D_vel": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Prior over velocity",
            },
            "F": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [1],
                "description": "Total Variational Free Energy (sum of factors)",
            },
            "G": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [1],
                "description": "Expected Free Energy",
            },
            "m_pos_to_vis": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Message: position→visual factor",
            },
            "m_prop_to_vel": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Message: proprioceptive factor→velocity",
            },
            "m_vel_to_prop": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Message: velocity→proprioceptive factor",
            },
            "m_vis_to_pos": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Message: visual factor→position",
            },
            "o_prop": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [4, 1],
                "description": "Proprioceptive observations (4D)",
            },
            "o_vis": {
                "type": "observation",
                "data_type": "integer",
                "dimensions": [6, 1],
                "description": "Visual observations (6 possible)",
            },
            "s_pos": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3, 1],
                "description": "Position state (3 discrete locations)",
            },
            "s_vel": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2, 1],
                "description": "Velocity state (2 levels: slow/fast)",
            },
            "t": {
                "type": "hidden_state",
                "data_type": "integer",
                "dimensions": [1],
            },
            "u": {
                "type": "action",
                "data_type": "integer",
                "dimensions": [1],
                "description": "Selected action",
            },
            "π": {
                "type": "policy",
                "data_type": "float",
                "dimensions": [2],
                "description": "Policy (2 actions: move/stay)",
            },
        }

        # Parameters
        self.parameters = {
            "A_prop": [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.5, 0.5]],
            "A_vis": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.05, 0.45, 0.5], [0.45, 0.05, 0.5], [0.5, 0.5, 0.0]],
            "B_pos": [[[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]], [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]],
            "B_vel": [[[0.8, 0.2], [0.2, 0.8]]],
            "C_prop": [[1.0, -1.0, 0.0, 0.0]],
            "C_vis": [[2.0, -0.5, -0.5, -0.5, -0.5, -0.5]],
            "D_pos": [[0.333, 0.333, 0.333]],
            "D_vel": [[0.5, 0.5]],
        }

# MODEL_DATA: {"model_name":"Factor Graph Active Inference Model","annotation":"A factor graph decomposition of an Active Inference generative model with:\n- Two independent observation modalities (visual and proprioceptive)\n- Two independent hidden state factors (position and velocity)\n- Factored joint distribution: P(o,s) = P(o_vis|s_pos) * P(o_prop|s_vel) * P(s_pos|s_vel) * P(s_vel)\n- Variable nodes: observation and state variables\n- Factor nodes: likelihood and transition factors\n- Enables modality-specific processing and efficient belief propagation","variables":[{"name":"o_vis","var_type":"observation","data_type":"integer","dimensions":[6,1]},{"name":"A_vis","var_type":"action","data_type":"float","dimensions":[6,3]},{"name":"o_prop","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"A_prop","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"s_pos","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"B_pos","var_type":"hidden_state","data_type":"float","dimensions":[3,3,2]},{"name":"s_vel","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"B_vel","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"D_pos","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_vel","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"C_vis","var_type":"hidden_state","data_type":"float","dimensions":[6]},{"name":"C_prop","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"m_pos_to_vis","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"m_vel_to_prop","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"m_vis_to_pos","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"m_prop_to_vel","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D_pos"],"target_variables":["s_pos"],"connection_type":"directed"},{"source_variables":["D_vel"],"target_variables":["s_vel"],"connection_type":"directed"},{"source_variables":["s_pos"],"target_variables":["A_vis"],"connection_type":"undirected"},{"source_variables":["A_vis"],"target_variables":["o_vis"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["A_prop"],"connection_type":"undirected"},{"source_variables":["A_prop"],"target_variables":["o_prop"],"connection_type":"undirected"},{"source_variables":["s_pos"],"target_variables":["B_pos"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["B_vel"],"connection_type":"undirected"},{"source_variables":["B_pos"],"target_variables":["s_pos"],"connection_type":"directed"},{"source_variables":["B_vel"],"target_variables":["s_vel"],"connection_type":"directed"},{"source_variables":["s_pos"],"target_variables":["m_pos_to_vis"],"connection_type":"undirected"},{"source_variables":["m_pos_to_vis"],"target_variables":["A_vis"],"connection_type":"undirected"},{"source_variables":["o_vis"],"target_variables":["m_vis_to_pos"],"connection_type":"undirected"},{"source_variables":["m_vis_to_pos"],"target_variables":["s_pos"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["m_vel_to_prop"],"connection_type":"undirected"},{"source_variables":["m_vel_to_prop"],"target_variables":["A_prop"],"connection_type":"undirected"},{"source_variables":["o_prop"],"target_variables":["m_prop_to_vel"],"connection_type":"undirected"},{"source_variables":["m_prop_to_vel"],"target_variables":["s_vel"],"connection_type":"undirected"},{"source_variables":["C_vis"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["C_prop"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B_pos"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["s_pos"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_vis"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_prop"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A_vis","value":[[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.05,0.45,0.5],[0.45,0.05,0.5],[0.5,0.5,0.0]],"param_type":"constant"},{"name":"A_prop","value":[[0.9,0.1],[0.1,0.9],[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"B_pos","value":[[[0.9,0.1,0.0],[0.0,0.9,0.1],[0.1,0.0,0.9]],[[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]]],"param_type":"constant"},{"name":"B_vel","value":[[[0.8,0.2],[0.2,0.8]]],"param_type":"constant"},{"name":"D_pos","value":[[0.333,0.333,0.333]],"param_type":"constant"},{"name":"D_vel","value":[[0.5,0.5]],"param_type":"constant"},{"name":"C_vis","value":[[2.0,-0.5,-0.5,-0.5,-0.5,-0.5]],"param_type":"constant"},{"name":"C_prop","value":[[1.0,-1.0,0.0,0.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A_vis","ontology_term":"VisualLikelihoodMatrix","description":null},{"variable_name":"A_prop","ontology_term":"ProprioceptiveLikelihoodMatrix","description":null},{"variable_name":"B_pos","ontology_term":"PositionTransitionMatrix","description":null},{"variable_name":"B_vel","ontology_term":"VelocityTransitionMatrix","description":null},{"variable_name":"D_pos","ontology_term":"PositionPrior","description":null},{"variable_name":"D_vel","ontology_term":"VelocityPrior","description":null},{"variable_name":"C_vis","ontology_term":"VisualPreferenceVector","description":null},{"variable_name":"C_prop","ontology_term":"ProprioceptivePreferenceVector","description":null},{"variable_name":"s_pos","ontology_term":"PositionHiddenState","description":null},{"variable_name":"s_vel","ontology_term":"VelocityHiddenState","description":null},{"variable_name":"o_vis","ontology_term":"VisualObservation","description":null},{"variable_name":"o_prop","ontology_term":"ProprioceptiveObservation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
