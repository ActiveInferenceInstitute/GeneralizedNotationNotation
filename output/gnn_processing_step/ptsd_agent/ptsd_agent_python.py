"""
GNN Model: PTSD Hierarchical Active Inference Agent
This hierarchical Active Inference agent consists of two levels:

- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages
- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors

The agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors.
Generated: 2025-07-25T22:44:10.760465
"""

import numpy as np
from typing import Dict, List, Any

class PTSDHierarchicalActiveInferenceAgentModel:
    """GNN Model: PTSD Hierarchical Active Inference Agent"""

    def __init__(self):
        self.model_name = "PTSD Hierarchical Active Inference Agent"
        self.version = "1.0"
        self.annotation = "This hierarchical Active Inference agent consists of two levels:

- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages
- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors

The agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors."

        # Variables
        self.variables = {
            "Advice": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3],
                "description": "Observation: blue, green, null",
            },
            "Affect": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2],
                "description": "Hidden state: angry, calm",
            },
            "AffectObs": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2],
                "description": "Observation from lower level",
            },
            "Arousal": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2],
                "description": "Observation: high, low",
            },
            "CardActions": {
                "type": "action",
                "data_type": "float",
                "dimensions": [3],
                "description": "Actions: blue, green, null",
            },
            "Choice": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Hidden state: blue, green, null",
            },
            "ChoiceObs": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [3],
                "description": "Observation: blue, green, null",
            },
            "ChoiceObsHigher": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [3],
                "description": "Observation from lower level",
            },
            "CorrectCard": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Hidden state: blue, green",
            },
            "CorrectCardObs": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [2],
                "description": "Observation from lower level",
            },
            "Feedback": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Observation: correct, incorrect, null",
            },
            "NullActions": {
                "type": "action",
                "data_type": "float",
                "dimensions": [1],
                "description": "Actions: NULL",
            },
            "NullActionsHigher": {
                "type": "action",
                "data_type": "float",
                "dimensions": [1],
                "description": "Actions: NULL",
            },
            "SafetyOther": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Hidden state: safe, danger",
            },
            "SafetySelf": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Hidden state: safe, danger",
            },
            "SafetyWorld": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Hidden state: safe, danger",
            },
            "Stage": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Hidden state: null, advice, decision",
            },
            "StageObs": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [3],
                "description": "Observation from lower level",
            },
            "TrustActions": {
                "type": "action",
                "data_type": "float",
                "dimensions": [2],
                "description": "Actions: trust, distrust",
            },
            "Trustworthiness": {
                "type": "hidden_state",
                "data_type": "float",
                "dimensions": [2],
                "description": "Hidden state: trust, distrust",
            },
            "TrustworthinessObs": {
                "type": "observation",
                "data_type": "float",
                "dimensions": [2],
                "description": "Observation from lower level",
            },
        }

        # Parameters
        self.parameters = {
            "A2_higher_0": [[0.667, 0.333], [0.333, 0.667]],
            "A2_higher_1": [[0.5, 0.5], [0.5, 0.5]],
            "A2_higher_2": [[0.333, 0.667], [0.667, 0.333]],
            "A2_higher_3": [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]],
            "A2_higher_4": [[0.333, 0.333, 0.333], [0.333, 0.333, 0.333]],
            "A_lower_0": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]],
            "A_lower_1": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]],
            "A_lower_2": [[1.0, 0.0], [0.0, 1.0]],
            "A_lower_3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "B2_higher_0": [[1.0, 0.0], [0.0, 1.0]],
            "B2_higher_1": [[1.0, 0.0], [0.0, 1.0]],
            "B2_higher_2": [[1.0, 0.0], [0.0, 1.0]],
            "B_lower_0": [[0.9, 0.1], [0.1, 0.9]],
            "B_lower_1": [[0.9, 0.1], [0.1, 0.9]],
            "B_lower_2": [[0.3333, 0.6667], [0.6667, 0.3333]],
            "B_lower_3": [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]],
            "B_lower_4": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            "C2_higher_0": [0.5, 0.5],
            "C2_higher_1": [0.5, 0.5],
            "C2_higher_2": [1.0, 0.0],
            "C2_higher_3": [0.333, 0.333, 0.333],
            "C2_higher_4": [0.333, 0.333, 0.333],
            "C_lower_0": [0.3333, 0.3333, 0.3333],
            "C_lower_1": [0.5, -3.5, 0.0],
            "C_lower_2": [0.65, 0.35],
            "C_lower_3": [0.3333, 0.3333, 0.3333],
            "D2_higher_0": [0.25, 0.75],
            "D2_higher_1": [0.25, 0.75],
            "D2_higher_2": [0.25, 0.75],
            "D_lower_0": [0.5, 0.5],
            "D_lower_1": [0.5, 0.5],
            "D_lower_2": [0.5, 0.5],
            "D_lower_3": [0.0, 0.0, 1.0],
            "D_lower_4": [1.0, 0.0, 0.0],
            "E2_higher": [0.0],
            "E_lower": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "affect_safety_association": 0.667,
            "alpha": 0.9,
            "alpha_policy": 16.0,
            "arousal_low_preference": 0.35,
            "cc": 0.5,
            "gamma": 16.0,
            "pA2_higher": 1.0,
            "pA_lower": 1.0,
            "pB2_higher": 1.0,
            "pB_lower": 1.0,
            "pD2_higher": 1.0,
            "pD_lower": 1.0,
            "p_Bchoice": 0.95,
            "p_Bcorrectcard": 0.9,
            "p_Bstage": 1.0,
            "p_Btrust": 0.9,
            "p_advice": 0.9,
            "prior_on_danger": 0.75,
            "trust_safety_association": 0.667,
        }

# MODEL_DATA: {"model_name":"PTSD Hierarchical Active Inference Agent","annotation":"This hierarchical Active Inference agent consists of two levels:\n\n- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages\n- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors\n\nThe agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors.","variables":[{"name":"Trustworthiness","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"CorrectCard","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"Affect","var_type":"action","data_type":"float","dimensions":[2]},{"name":"Choice","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"Stage","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"Advice","var_type":"action","data_type":"float","dimensions":[3]},{"name":"Feedback","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"Arousal","var_type":"action","data_type":"float","dimensions":[2]},{"name":"ChoiceObs","var_type":"observation","data_type":"float","dimensions":[3]},{"name":"TrustActions","var_type":"action","data_type":"float","dimensions":[2]},{"name":"CardActions","var_type":"action","data_type":"float","dimensions":[3]},{"name":"NullActions","var_type":"action","data_type":"float","dimensions":[1]},{"name":"SafetySelf","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"SafetyWorld","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"SafetyOther","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"TrustworthinessObs","var_type":"observation","data_type":"float","dimensions":[2]},{"name":"CorrectCardObs","var_type":"observation","data_type":"float","dimensions":[2]},{"name":"AffectObs","var_type":"action","data_type":"float","dimensions":[2]},{"name":"ChoiceObsHigher","var_type":"observation","data_type":"float","dimensions":[3]},{"name":"StageObs","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"NullActionsHigher","var_type":"action","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["Trustworthiness"],"target_variables":["Advice"],"connection_type":"directed"},{"source_variables":["CorrectCard"],"target_variables":["Feedback"],"connection_type":"directed"},{"source_variables":["Affect"],"target_variables":["Arousal"],"connection_type":"directed"},{"source_variables":["Choice"],"target_variables":["ChoiceObs"],"connection_type":"directed"},{"source_variables":["Stage"],"target_variables":["Advice"],"connection_type":"directed"},{"source_variables":["SafetySelf"],"target_variables":["TrustworthinessObs"],"connection_type":"directed"},{"source_variables":["SafetyWorld"],"target_variables":["CorrectCardObs"],"connection_type":"directed"},{"source_variables":["SafetyOther"],"target_variables":["AffectObs"],"connection_type":"directed"},{"source_variables":["Trustworthiness"],"target_variables":["TrustworthinessObs"],"connection_type":"directed"},{"source_variables":["CorrectCard"],"target_variables":["CorrectCardObs"],"connection_type":"directed"},{"source_variables":["Affect"],"target_variables":["AffectObs"],"connection_type":"directed"},{"source_variables":["Choice"],"target_variables":["ChoiceObsHigher"],"connection_type":"directed"},{"source_variables":["Stage"],"target_variables":["StageObs"],"connection_type":"directed"}],"parameters":[{"name":"A_lower_0","value":[[0.9,0.1,0.0],[0.1,0.9,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"A_lower_1","value":[[0.9,0.1,0.0],[0.1,0.9,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"A_lower_2","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"A_lower_3","value":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"B_lower_0","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B_lower_1","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B_lower_2","value":[[0.3333,0.6667],[0.6667,0.3333]],"param_type":"constant"},{"name":"B_lower_3","value":[[0.95,0.025,0.025],[0.025,0.95,0.025],[0.025,0.025,0.95]],"param_type":"constant"},{"name":"B_lower_4","value":[[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0]],"param_type":"constant"},{"name":"C_lower_0","value":[0.3333,0.3333,0.3333],"param_type":"constant"},{"name":"C_lower_1","value":[0.5,-3.5,0.0],"param_type":"constant"},{"name":"C_lower_2","value":[0.65,0.35],"param_type":"constant"},{"name":"C_lower_3","value":[0.3333,0.3333,0.3333],"param_type":"constant"},{"name":"D_lower_0","value":[0.5,0.5],"param_type":"constant"},{"name":"D_lower_1","value":[0.5,0.5],"param_type":"constant"},{"name":"D_lower_2","value":[0.5,0.5],"param_type":"constant"},{"name":"D_lower_3","value":[0.0,0.0,1.0],"param_type":"constant"},{"name":"D_lower_4","value":[1.0,0.0,0.0],"param_type":"constant"},{"name":"E_lower","value":[0.0,0.0,0.0,0.0,0.0,0.0],"param_type":"constant"},{"name":"A2_higher_0","value":[[0.667,0.333],[0.333,0.667]],"param_type":"constant"},{"name":"A2_higher_1","value":[[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"A2_higher_2","value":[[0.333,0.667],[0.667,0.333]],"param_type":"constant"},{"name":"A2_higher_3","value":[[0.333,0.333,0.333],[0.333,0.333,0.333]],"param_type":"constant"},{"name":"A2_higher_4","value":[[0.333,0.333,0.333],[0.333,0.333,0.333]],"param_type":"constant"},{"name":"B2_higher_0","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"B2_higher_1","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"B2_higher_2","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"C2_higher_0","value":[0.5,0.5],"param_type":"constant"},{"name":"C2_higher_1","value":[0.5,0.5],"param_type":"constant"},{"name":"C2_higher_2","value":[1.0,0.0],"param_type":"constant"},{"name":"C2_higher_3","value":[0.333,0.333,0.333],"param_type":"constant"},{"name":"C2_higher_4","value":[0.333,0.333,0.333],"param_type":"constant"},{"name":"D2_higher_0","value":[0.25,0.75],"param_type":"constant"},{"name":"D2_higher_1","value":[0.25,0.75],"param_type":"constant"},{"name":"D2_higher_2","value":[0.25,0.75],"param_type":"constant"},{"name":"E2_higher","value":[0.0],"param_type":"constant"},{"name":"pA_lower","value":1.0,"param_type":"constant"},{"name":"pB_lower","value":1.0,"param_type":"constant"},{"name":"pD_lower","value":1.0,"param_type":"constant"},{"name":"pA2_higher","value":1.0,"param_type":"constant"},{"name":"pB2_higher","value":1.0,"param_type":"constant"},{"name":"pD2_higher","value":1.0,"param_type":"constant"},{"name":"p_advice","value":0.9,"param_type":"constant"},{"name":"alpha","value":0.9,"param_type":"constant"},{"name":"p_Btrust","value":0.9,"param_type":"constant"},{"name":"p_Bcorrectcard","value":0.9,"param_type":"constant"},{"name":"p_Bchoice","value":0.95,"param_type":"constant"},{"name":"p_Bstage","value":1.0,"param_type":"constant"},{"name":"cc","value":0.5,"param_type":"constant"},{"name":"arousal_low_preference","value":0.35,"param_type":"constant"},{"name":"trust_safety_association","value":0.667,"param_type":"constant"},{"name":"affect_safety_association","value":0.667,"param_type":"constant"},{"name":"prior_on_danger","value":0.75,"param_type":"constant"},{"name":"gamma","value":16.0,"param_type":"constant"},{"name":"alpha_policy","value":16.0,"param_type":"constant"}],"equations":["Equation(node_type='Equation', source_location=None, metadata={}, id='271d22f5-f15f-4d56-be85-53ba157e29b2', label=None, content='F = D_KL[Q(s)||P(s|o)] - ln P(o)                                   # Free Energy Q(s) = softmax(ln A + ln B + ln C + ln D)                          # Variational Message Passing \u03c0* = argmin_\u03c0 F(\u03c0)                                                  # Policy Selection', format='latex', description=None)","Equation(node_type='Equation', source_location=None, metadata={}, id='054be701-12e4-4fd6-b562-fe833f344316', label=None, content='F_high = D_KL[Q(s_high)||P(s_high|o_high)] - ln P(o_high)         # Free Energy Q(s_high) = softmax(ln A_high + ln B_high + ln C_high + ln D_high) # Hierarchical Inference o_high = f(Q_low(s_low))                                            # Inter-level Coupling', format='latex', description=None)"],"time_specification":{"time_type":"Static","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[]}
