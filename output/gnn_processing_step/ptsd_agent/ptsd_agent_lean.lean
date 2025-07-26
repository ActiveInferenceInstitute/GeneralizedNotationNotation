-- GNN Model in Lean 4
-- Model: PTSD Hierarchical Active Inference Agent
-- This hierarchical Active Inference agent consists of two levels:

- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages
- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors

The agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors.

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

namespace PTSDHierarchicalActiveInferenceAgent

-- Variables
variable (Advice : ℝ)
variable (Affect : ℝ)
variable (AffectObs : ℝ)
variable (Arousal : ℝ)
variable (CardActions : ℝ)
variable (Choice : ℝ)
variable (ChoiceObs : ℝ)
variable (ChoiceObsHigher : ℝ)
variable (CorrectCard : ℝ)
variable (CorrectCardObs : ℝ)
variable (Feedback : ℝ)
variable (NullActions : ℝ)
variable (NullActionsHigher : ℝ)
variable (SafetyOther : ℝ)
variable (SafetySelf : ℝ)
variable (SafetyWorld : ℝ)
variable (Stage : ℝ)
variable (StageObs : ℝ)
variable (TrustActions : ℝ)
variable (Trustworthiness : ℝ)
variable (TrustworthinessObs : ℝ)

structure PTSDHierarchicalActiveInferenceAgentModel where
  Advice : ℝ
  Affect : ℝ
  AffectObs : ℝ
  Arousal : ℝ
  CardActions : ℝ
  Choice : ℝ
  ChoiceObs : ℝ
  ChoiceObsHigher : ℝ
  CorrectCard : ℝ
  CorrectCardObs : ℝ
  Feedback : ℝ
  NullActions : ℝ
  NullActionsHigher : ℝ
  SafetyOther : ℝ
  SafetySelf : ℝ
  SafetyWorld : ℝ
  Stage : ℝ
  StageObs : ℝ
  TrustActions : ℝ
  Trustworthiness : ℝ
  TrustworthinessObs : ℝ

end PTSDHierarchicalActiveInferenceAgent
-- MODEL_DATA: {"model_name":"PTSD Hierarchical Active Inference Agent","annotation":"This hierarchical Active Inference agent consists of two levels:\n\n- **Lower Level Agent**: Processes sensorimotor information including trustworthiness, card states, affect, choices, and game stages\n- **Higher Level Agent**: Processes abstract safety concepts (self, world, other) based on lower-level posteriors\n\nThe agents communicate bidirectionally: lower-level posteriors become higher-level observations, and higher-level inferred states become lower-level priors.","variables":[{"name":"Trustworthiness","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"CorrectCard","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"Affect","var_type":"action","data_type":"float","dimensions":[2]},{"name":"Choice","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"Stage","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"Advice","var_type":"action","data_type":"float","dimensions":[3]},{"name":"Feedback","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"Arousal","var_type":"action","data_type":"float","dimensions":[2]},{"name":"ChoiceObs","var_type":"observation","data_type":"float","dimensions":[3]},{"name":"TrustActions","var_type":"action","data_type":"float","dimensions":[2]},{"name":"CardActions","var_type":"action","data_type":"float","dimensions":[3]},{"name":"NullActions","var_type":"action","data_type":"float","dimensions":[1]},{"name":"SafetySelf","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"SafetyWorld","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"SafetyOther","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"TrustworthinessObs","var_type":"observation","data_type":"float","dimensions":[2]},{"name":"CorrectCardObs","var_type":"observation","data_type":"float","dimensions":[2]},{"name":"AffectObs","var_type":"action","data_type":"float","dimensions":[2]},{"name":"ChoiceObsHigher","var_type":"observation","data_type":"float","dimensions":[3]},{"name":"StageObs","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"NullActionsHigher","var_type":"action","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["Trustworthiness"],"target_variables":["Advice"],"connection_type":"directed"},{"source_variables":["CorrectCard"],"target_variables":["Feedback"],"connection_type":"directed"},{"source_variables":["Affect"],"target_variables":["Arousal"],"connection_type":"directed"},{"source_variables":["Choice"],"target_variables":["ChoiceObs"],"connection_type":"directed"},{"source_variables":["Stage"],"target_variables":["Advice"],"connection_type":"directed"},{"source_variables":["SafetySelf"],"target_variables":["TrustworthinessObs"],"connection_type":"directed"},{"source_variables":["SafetyWorld"],"target_variables":["CorrectCardObs"],"connection_type":"directed"},{"source_variables":["SafetyOther"],"target_variables":["AffectObs"],"connection_type":"directed"},{"source_variables":["Trustworthiness"],"target_variables":["TrustworthinessObs"],"connection_type":"directed"},{"source_variables":["CorrectCard"],"target_variables":["CorrectCardObs"],"connection_type":"directed"},{"source_variables":["Affect"],"target_variables":["AffectObs"],"connection_type":"directed"},{"source_variables":["Choice"],"target_variables":["ChoiceObsHigher"],"connection_type":"directed"},{"source_variables":["Stage"],"target_variables":["StageObs"],"connection_type":"directed"}],"parameters":[{"name":"A_lower_0","value":[[0.9,0.1,0.0],[0.1,0.9,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"A_lower_1","value":[[0.9,0.1,0.0],[0.1,0.9,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"A_lower_2","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"A_lower_3","value":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"B_lower_0","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B_lower_1","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B_lower_2","value":[[0.3333,0.6667],[0.6667,0.3333]],"param_type":"constant"},{"name":"B_lower_3","value":[[0.95,0.025,0.025],[0.025,0.95,0.025],[0.025,0.025,0.95]],"param_type":"constant"},{"name":"B_lower_4","value":[[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0]],"param_type":"constant"},{"name":"C_lower_0","value":[0.3333,0.3333,0.3333],"param_type":"constant"},{"name":"C_lower_1","value":[0.5,-3.5,0.0],"param_type":"constant"},{"name":"C_lower_2","value":[0.65,0.35],"param_type":"constant"},{"name":"C_lower_3","value":[0.3333,0.3333,0.3333],"param_type":"constant"},{"name":"D_lower_0","value":[0.5,0.5],"param_type":"constant"},{"name":"D_lower_1","value":[0.5,0.5],"param_type":"constant"},{"name":"D_lower_2","value":[0.5,0.5],"param_type":"constant"},{"name":"D_lower_3","value":[0.0,0.0,1.0],"param_type":"constant"},{"name":"D_lower_4","value":[1.0,0.0,0.0],"param_type":"constant"},{"name":"E_lower","value":[0.0,0.0,0.0,0.0,0.0,0.0],"param_type":"constant"},{"name":"A2_higher_0","value":[[0.667,0.333],[0.333,0.667]],"param_type":"constant"},{"name":"A2_higher_1","value":[[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"A2_higher_2","value":[[0.333,0.667],[0.667,0.333]],"param_type":"constant"},{"name":"A2_higher_3","value":[[0.333,0.333,0.333],[0.333,0.333,0.333]],"param_type":"constant"},{"name":"A2_higher_4","value":[[0.333,0.333,0.333],[0.333,0.333,0.333]],"param_type":"constant"},{"name":"B2_higher_0","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"B2_higher_1","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"B2_higher_2","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"C2_higher_0","value":[0.5,0.5],"param_type":"constant"},{"name":"C2_higher_1","value":[0.5,0.5],"param_type":"constant"},{"name":"C2_higher_2","value":[1.0,0.0],"param_type":"constant"},{"name":"C2_higher_3","value":[0.333,0.333,0.333],"param_type":"constant"},{"name":"C2_higher_4","value":[0.333,0.333,0.333],"param_type":"constant"},{"name":"D2_higher_0","value":[0.25,0.75],"param_type":"constant"},{"name":"D2_higher_1","value":[0.25,0.75],"param_type":"constant"},{"name":"D2_higher_2","value":[0.25,0.75],"param_type":"constant"},{"name":"E2_higher","value":[0.0],"param_type":"constant"},{"name":"pA_lower","value":1.0,"param_type":"constant"},{"name":"pB_lower","value":1.0,"param_type":"constant"},{"name":"pD_lower","value":1.0,"param_type":"constant"},{"name":"pA2_higher","value":1.0,"param_type":"constant"},{"name":"pB2_higher","value":1.0,"param_type":"constant"},{"name":"pD2_higher","value":1.0,"param_type":"constant"},{"name":"p_advice","value":0.9,"param_type":"constant"},{"name":"alpha","value":0.9,"param_type":"constant"},{"name":"p_Btrust","value":0.9,"param_type":"constant"},{"name":"p_Bcorrectcard","value":0.9,"param_type":"constant"},{"name":"p_Bchoice","value":0.95,"param_type":"constant"},{"name":"p_Bstage","value":1.0,"param_type":"constant"},{"name":"cc","value":0.5,"param_type":"constant"},{"name":"arousal_low_preference","value":0.35,"param_type":"constant"},{"name":"trust_safety_association","value":0.667,"param_type":"constant"},{"name":"affect_safety_association","value":0.667,"param_type":"constant"},{"name":"prior_on_danger","value":0.75,"param_type":"constant"},{"name":"gamma","value":16.0,"param_type":"constant"},{"name":"alpha_policy","value":16.0,"param_type":"constant"}],"equations":["Equation(node_type='Equation', source_location=None, metadata={}, id='271d22f5-f15f-4d56-be85-53ba157e29b2', label=None, content='F = D_KL[Q(s)||P(s|o)] - ln P(o)                                   # Free Energy Q(s) = softmax(ln A + ln B + ln C + ln D)                          # Variational Message Passing \u03c0* = argmin_\u03c0 F(\u03c0)                                                  # Policy Selection', format='latex', description=None)","Equation(node_type='Equation', source_location=None, metadata={}, id='054be701-12e4-4fd6-b562-fe833f344316', label=None, content='F_high = D_KL[Q(s_high)||P(s_high|o_high)] - ln P(o_high)         # Free Energy Q(s_high) = softmax(ln A_high + ln B_high + ln C_high + ln D_high) # Hierarchical Inference o_high = f(Q_low(s_low))                                            # Inter-level Coupling', format='latex', description=None)"],"time_specification":{"time_type":"Static","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[]}
