module HierarchicalActiveInferencePOMDP where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A_level1 = A_level1 Double
data A_level2 = A_level2 Double
data B_level1 = B_level1 Double
data B_level2 = B_level2 Double
data C_level1 = C_level1 Double
data C_level2 = C_level2 Double
data D_level1 = D_level1 Double
data D_level2 = D_level2 Double
data G1 = G1 Double
data G2 = G2 Double
data o_level1 = o_level1 Int
data o_level2 = o_level2 Double
data s_level1 = s_level1 Double
data s_level2 = s_level2 Double
data t1 = t1 Int
data t2 = t2 Int
data u_level1 = u_level1 Int
data x_next1 = x_next1 Double
data π1 = π1 Double

-- Connections as Functions
D_level1Tos_level1 :: D_level1 -> s_level1
D_level1Tos_level1 x = undefined
s_level1ToA_level1 :: s_level1 -> A_level1
s_level1ToA_level1 x = undefined
s_level1Tox_next1 :: s_level1 -> x_next1
s_level1Tox_next1 x = undefined
A_level1Too_level1 :: A_level1 -> o_level1
A_level1Too_level1 x = undefined
C_level1ToG1 :: C_level1 -> G1
C_level1ToG1 x = undefined
G1Toπ1 :: G1 -> π1
G1Toπ1 x = undefined
π1Tou_level1 :: π1 -> u_level1
π1Tou_level1 x = undefined
B_level1Tou_level1 :: B_level1 -> u_level1
B_level1Tou_level1 x = undefined
u_level1Tox_next1 :: u_level1 -> x_next1
u_level1Tox_next1 x = undefined
s_level1Too_level2 :: s_level1 -> o_level2
s_level1Too_level2 x = undefined
D_level2Tos_level2 :: D_level2 -> s_level2
D_level2Tos_level2 x = undefined
s_level2ToA_level2 :: s_level2 -> A_level2
s_level2ToA_level2 x = undefined
A_level2ToD_level1 :: A_level2 -> D_level1
A_level2ToD_level1 x = undefined
s_level2ToB_level2 :: s_level2 -> B_level2
s_level2ToB_level2 x = undefined
C_level2ToG2 :: C_level2 -> G2
C_level2ToG2 x = undefined
G2Tos_level2 :: G2 -> s_level2
G2Tos_level2 x = undefined

-- MODEL_DATA: {"model_name":"Hierarchical Active Inference POMDP","annotation":"A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1","variables":[{"name":"A_level1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_level1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_level1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_level1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u_level1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level2","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"B_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"C_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"D_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_level2","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t1","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"t2","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_level1"],"target_variables":["s_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["A_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["A_level1"],"target_variables":["o_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c01"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_level1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["o_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level2"],"target_variables":["s_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["A_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level2"],"target_variables":["D_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["B_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["s_level2"],"connection_type":"directed"}],"parameters":[{"name":"A_level1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"B_level1","value":[[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]],[[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C_level1","value":[[0.1,0.1,0.1,1.0]],"param_type":"constant"},{"name":"D_level1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A_level2","value":[[0.9,0.1],[0.1,0.9],[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"B_level2","value":[[[0.9,0.1],[0.1,0.9]]],"param_type":"constant"},{"name":"C_level2","value":[[0.0,0.5,0.0,0.5]],"param_type":"constant"},{"name":"D_level2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":8,"param_type":"constant"},{"name":"num_obs","value":16,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_timesteps","value":20,"param_type":"constant"},{"name":"num_hidden_states_l1","value":4,"param_type":"constant"},{"name":"num_obs_l1","value":4,"param_type":"constant"},{"name":"num_actions_l1","value":3,"param_type":"constant"},{"name":"num_context_states_l2","value":2,"param_type":"constant"},{"name":"timescale_ratio","value":5,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A_level1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_level1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_level1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_level1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_level1","ontology_term":"HiddenState","description":null},{"variable_name":"o_level1","ontology_term":"Observation","description":null},{"variable_name":"\u03c01","ontology_term":"PolicyVector","description":null},{"variable_name":"u_level1","ontology_term":"Action","description":null},{"variable_name":"G1","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"A_level2","ontology_term":"HigherLevelLikelihoodMatrix","description":null},{"variable_name":"B_level2","ontology_term":"ContextTransitionMatrix","description":null},{"variable_name":"s_level2","ontology_term":"ContextualHiddenState","description":null},{"variable_name":"o_level2","ontology_term":"HigherLevelObservation","description":null},{"variable_name":"G2","ontology_term":"HigherLevelExpectedFreeEnergy","description":null}]}
