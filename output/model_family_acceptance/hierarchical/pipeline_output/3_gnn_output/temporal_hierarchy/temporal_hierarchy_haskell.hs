module ThreeLevelTemporalHierarchyAgent where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A_level0 = A_level0 Double
data A_level1 = A_level1 Double
data A_level2 = A_level2 Double
data B_level0 = B_level0 Double
data B_level1 = B_level1 Double
data B_level2 = B_level2 Double
data C_level0 = C_level0 Double
data C_level1 = C_level1 Double
data C_level2 = C_level2 Double
data D_level0 = D_level0 Double
data D_level1 = D_level1 Double
data D_level2 = D_level2 Double
data G0 = G0 Double
data G1 = G1 Double
data G2 = G2 Double
data o_level0 = o_level0 Int
data o_level1 = o_level1 Double
data o_level2 = o_level2 Double
data pi0 = pi0 Double
data pi1 = pi1 Double
data pi2 = pi2 Double
data s_level0 = s_level0 Double
data s_level1 = s_level1 Double
data s_level2 = s_level2 Double
data t = t Int
data tau_level0 = tau_level0 Double
data tau_level1 = tau_level1 Double
data tau_level2 = tau_level2 Double
data u_level0 = u_level0 Int
data u_level1 = u_level1 Int
data u_level2 = u_level2 Int

-- Connections as Functions
D_level0Tos_level0 :: D_level0 -> s_level0
D_level0Tos_level0 x = undefined
s_level0ToA_level0 :: s_level0 -> A_level0
s_level0ToA_level0 x = undefined
A_level0Too_level0 :: A_level0 -> o_level0
A_level0Too_level0 x = undefined
C_level0ToG0 :: C_level0 -> G0
C_level0ToG0 x = undefined
G0Topi0 :: G0 -> pi0
G0Topi0 x = undefined
pi0Tou_level0 :: pi0 -> u_level0
pi0Tou_level0 x = undefined
B_level0Tou_level0 :: B_level0 -> u_level0
B_level0Tou_level0 x = undefined
D_level1Tos_level1 :: D_level1 -> s_level1
D_level1Tos_level1 x = undefined
s_level1ToA_level1 :: s_level1 -> A_level1
s_level1ToA_level1 x = undefined
A_level1Too_level1 :: A_level1 -> o_level1
A_level1Too_level1 x = undefined
C_level1ToG1 :: C_level1 -> G1
C_level1ToG1 x = undefined
G1Topi1 :: G1 -> pi1
G1Topi1 x = undefined
pi1Tou_level1 :: pi1 -> u_level1
pi1Tou_level1 x = undefined
B_level1Tou_level1 :: B_level1 -> u_level1
B_level1Tou_level1 x = undefined
D_level2Tos_level2 :: D_level2 -> s_level2
D_level2Tos_level2 x = undefined
s_level2ToA_level2 :: s_level2 -> A_level2
s_level2ToA_level2 x = undefined
A_level2Too_level2 :: A_level2 -> o_level2
A_level2Too_level2 x = undefined
C_level2ToG2 :: C_level2 -> G2
C_level2ToG2 x = undefined
G2Topi2 :: G2 -> pi2
G2Topi2 x = undefined
pi2Tou_level2 :: pi2 -> u_level2
pi2Tou_level2 x = undefined
B_level2Tou_level2 :: B_level2 -> u_level2
B_level2Tou_level2 x = undefined
s_level2ToC_level1 :: s_level2 -> C_level1
s_level2ToC_level1 x = undefined
s_level1ToC_level0 :: s_level1 -> C_level0
s_level1ToC_level0 x = undefined
s_level2ToD_level1 :: s_level2 -> D_level1
s_level2ToD_level1 x = undefined
s_level0Too_level1 :: s_level0 -> o_level1
s_level0Too_level1 x = undefined
s_level1Too_level2 :: s_level1 -> o_level2
s_level1Too_level2 x = undefined

-- MODEL_DATA: {"model_name":"Three-Level Temporal Hierarchy Agent","annotation":"A three-level hierarchical Active Inference agent with distinct temporal scales:\n\n- Level 0 (fast, 100ms): Sensorimotor control \u2014 immediate reflexive responses\n- Level 1 (medium, 1s): Tactical planning \u2014 goal-directed behavior sequences\n- Level 2 (slow, 10s): Strategic planning \u2014 long-term objective management\n- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences\n- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy\n- Each level maintains its own generative model with A, B, C, D matrices\n- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)\n- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference","variables":[{"name":"A_level0","var_type":"action","data_type":"float","dimensions":[3,4]},{"name":"B_level0","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_level0","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_level0","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_level0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_level0","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"pi0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u_level0","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level1","var_type":"action","data_type":"float","dimensions":[4,3]},{"name":"B_level1","var_type":"hidden_state","data_type":"float","dimensions":[3,3,3]},{"name":"C_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_level1","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"s_level1","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o_level1","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u_level1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level2","var_type":"action","data_type":"float","dimensions":[3,2]},{"name":"B_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,2]},{"name":"C_level2","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_level2","var_type":"observation","data_type":"float","dimensions":[3,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u_level2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau_level0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau_level1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau_level2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_level0"],"target_variables":["s_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level0"],"target_variables":["A_level0"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level0"],"target_variables":["o_level0"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level0"],"target_variables":["G0"],"connection_type":"directed"},{"annotation":null,"source_variables":["G0"],"target_variables":["pi0"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi0"],"target_variables":["u_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level0"],"target_variables":["u_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level1"],"target_variables":["s_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["A_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level1"],"target_variables":["o_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level2"],"target_variables":["s_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["A_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level2"],"target_variables":["o_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi2"],"target_variables":["u_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level2"],"target_variables":["u_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["C_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["C_level0"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["D_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level0"],"target_variables":["o_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["o_level2"],"connection_type":"directed"}],"parameters":[{"name":"A_level0","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05]],"param_type":"constant"},{"name":"C_level0","value":[[0.0,-1.0,1.0]],"param_type":"constant"},{"name":"D_level0","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A_level1","value":[[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.1,0.1,0.1]],"param_type":"constant"},{"name":"C_level1","value":[[-0.5,1.0,1.5,-1.0]],"param_type":"constant"},{"name":"D_level1","value":[[0.33,0.33,0.34]],"param_type":"constant"},{"name":"A_level2","value":[[0.9,0.1],[0.1,0.9],[0.1,0.1]],"param_type":"constant"},{"name":"C_level2","value":[[-1.0,2.0,0.5]],"param_type":"constant"},{"name":"D_level2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"tau_level0","value":[[0.1]],"param_type":"constant"},{"name":"tau_level1","value":[[1.0]],"param_type":"constant"},{"name":"tau_level2","value":[[10.0]],"param_type":"constant"},{"name":"B_level0","value":[[[0.9,0.05,0.05,0.0],[0.05,0.9,0.05,0.0],[0.05,0.05,0.9,0.0],[0.0,0.0,0.0,1.0]],[[0.05,0.9,0.05,0.0],[0.9,0.05,0.05,0.0],[0.05,0.05,0.9,0.0],[0.0,0.0,0.0,1.0]],[[0.9,0.05,0.05,0.0],[0.05,0.9,0.05,0.0],[0.05,0.05,0.9,0.0],[0.0,0.0,0.0,1.0]]],"param_type":"constant"},{"name":"B_level1","value":[[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],[[0.05,0.9,0.05],[0.9,0.05,0.05],[0.05,0.05,0.9]],[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]],"param_type":"constant"},{"name":"B_level2","value":[[[0.95,0.05],[0.05,0.95]]],"param_type":"constant"},{"name":"num_hidden_states","value":24,"param_type":"constant"},{"name":"num_obs","value":36,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_levels","value":3,"param_type":"constant"},{"name":"num_states_l0","value":4,"param_type":"constant"},{"name":"num_obs_l0","value":3,"param_type":"constant"},{"name":"num_actions_l0","value":3,"param_type":"constant"},{"name":"num_states_l1","value":3,"param_type":"constant"},{"name":"num_obs_l1","value":4,"param_type":"constant"},{"name":"num_actions_l1","value":3,"param_type":"constant"},{"name":"num_states_l2","value":2,"param_type":"constant"},{"name":"num_obs_l2","value":3,"param_type":"constant"},{"name":"num_actions_l2","value":2,"param_type":"constant"},{"name":"timescale_ratio_1_0","value":10,"param_type":"constant"},{"name":"timescale_ratio_2_1","value":10,"param_type":"constant"},{"name":"num_timesteps","value":100,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":100,"step_size":null},"ontology_mappings":[{"variable_name":"A_level0","ontology_term":"FastLikelihoodMatrix","description":null},{"variable_name":"B_level0","ontology_term":"FastTransitionMatrix","description":null},{"variable_name":"C_level0","ontology_term":"FastPreferenceVector","description":null},{"variable_name":"D_level0","ontology_term":"FastPrior","description":null},{"variable_name":"s_level0","ontology_term":"FastHiddenState","description":null},{"variable_name":"o_level0","ontology_term":"FastObservation","description":null},{"variable_name":"pi0","ontology_term":"FastPolicyVector","description":null},{"variable_name":"u_level0","ontology_term":"FastAction","description":null},{"variable_name":"G0","ontology_term":"FastExpectedFreeEnergy","description":null},{"variable_name":"A_level1","ontology_term":"TacticalLikelihoodMatrix","description":null},{"variable_name":"B_level1","ontology_term":"TacticalTransitionMatrix","description":null},{"variable_name":"C_level1","ontology_term":"TacticalPreferenceVector","description":null},{"variable_name":"D_level1","ontology_term":"TacticalPrior","description":null},{"variable_name":"s_level1","ontology_term":"TacticalHiddenState","description":null},{"variable_name":"o_level1","ontology_term":"TacticalObservation","description":null},{"variable_name":"pi1","ontology_term":"TacticalPolicyVector","description":null},{"variable_name":"u_level1","ontology_term":"TacticalAction","description":null},{"variable_name":"G1","ontology_term":"TacticalExpectedFreeEnergy","description":null},{"variable_name":"A_level2","ontology_term":"StrategicLikelihoodMatrix","description":null},{"variable_name":"B_level2","ontology_term":"StrategicTransitionMatrix","description":null},{"variable_name":"C_level2","ontology_term":"StrategicPreferenceVector","description":null},{"variable_name":"D_level2","ontology_term":"StrategicPrior","description":null},{"variable_name":"s_level2","ontology_term":"StrategicHiddenState","description":null},{"variable_name":"o_level2","ontology_term":"StrategicObservation","description":null},{"variable_name":"pi2","ontology_term":"StrategicPolicyVector","description":null},{"variable_name":"u_level2","ontology_term":"StrategicAction","description":null},{"variable_name":"G2","ontology_term":"StrategicExpectedFreeEnergy","description":null},{"variable_name":"tau_level0","ontology_term":"FastTimeConstant","description":null},{"variable_name":"tau_level1","ontology_term":"TacticalTimeConstant","description":null},{"variable_name":"tau_level2","ontology_term":"StrategicTimeConstant","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
