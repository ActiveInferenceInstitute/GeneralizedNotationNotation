module MultiAgentCooperativeActiveInference where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A_agent1 = A_agent1 Double
data A_agent2 = A_agent2 Double
data B_agent1 = B_agent1 Double
data B_agent2 = B_agent2 Double
data C_agent1 = C_agent1 Double
data C_agent2 = C_agent2 Double
data D_agent1 = D_agent1 Double
data D_agent2 = D_agent2 Double
data G1 = G1 Double
data G2 = G2 Double
data o_agent1 = o_agent1 Int
data o_agent2 = o_agent2 Int
data o_joint = o_joint Int
data s_agent1 = s_agent1 Double
data s_agent2 = s_agent2 Double
data s_joint = s_joint Double
data t = t Int
data u1 = u1 Int
data u2 = u2 Int
data x_next1 = x_next1 Double
data x_next2 = x_next2 Double
data π1 = π1 Double
data π2 = π2 Double

-- Connections as Functions
D_agent1Tos_agent1 :: D_agent1 -> s_agent1
D_agent1Tos_agent1 x = undefined
s_agent1ToA_agent1 :: s_agent1 -> A_agent1
s_agent1ToA_agent1 x = undefined
A_agent1Too_agent1 :: A_agent1 -> o_agent1
A_agent1Too_agent1 x = undefined
s_agent1Tox_next1 :: s_agent1 -> x_next1
s_agent1Tox_next1 x = undefined
C_agent1ToG1 :: C_agent1 -> G1
C_agent1ToG1 x = undefined
G1Toπ1 :: G1 -> π1
G1Toπ1 x = undefined
π1Tou1 :: π1 -> u1
π1Tou1 x = undefined
B_agent1Tou1 :: B_agent1 -> u1
B_agent1Tou1 x = undefined
D_agent2Tos_agent2 :: D_agent2 -> s_agent2
D_agent2Tos_agent2 x = undefined
s_agent2ToA_agent2 :: s_agent2 -> A_agent2
s_agent2ToA_agent2 x = undefined
A_agent2Too_agent2 :: A_agent2 -> o_agent2
A_agent2Too_agent2 x = undefined
s_agent2Tox_next2 :: s_agent2 -> x_next2
s_agent2Tox_next2 x = undefined
C_agent2ToG2 :: C_agent2 -> G2
C_agent2ToG2 x = undefined
G2Toπ2 :: G2 -> π2
G2Toπ2 x = undefined
π2Tou2 :: π2 -> u2
π2Tou2 x = undefined
B_agent2Tou2 :: B_agent2 -> u2
B_agent2Tou2 x = undefined
u1Tos_joint :: u1 -> s_joint
u1Tos_joint x = undefined
u2Tos_joint :: u2 -> s_joint
u2Tos_joint x = undefined
s_jointToo_joint :: s_joint -> o_joint
s_jointToo_joint x = undefined
o_agent1Tos_joint :: o_agent1 -> s_joint
o_agent1Tos_joint x = undefined
o_agent2Tos_joint :: o_agent2 -> s_joint
o_agent2Tos_joint x = undefined

-- MODEL_DATA: {"model_name":"Multi-Agent Cooperative Active Inference","annotation":"Two Active Inference agents cooperating on a joint task:\n\n- Agent 1 and Agent 2 each maintain independent beliefs\n- Shared observation space: agents observe each other's actions\n- Joint task state includes both agents' positions (4x4 = 16 joint states)\n- Cooperative preferences: both agents prefer the same goal configuration\n- Models social cognition and coordination without explicit communication","variables":[{"name":"A_agent1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_agent1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_agent2","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_agent2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c02","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"s_joint","var_type":"hidden_state","data_type":"float","dimensions":[16,1]},{"name":"o_joint","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_agent1"],"target_variables":["s_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent1"],"target_variables":["A_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent1"],"target_variables":["o_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_agent1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["C_agent1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c01"],"target_variables":["u1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent1"],"target_variables":["u1"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_agent2"],"target_variables":["s_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent2"],"target_variables":["A_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent2"],"target_variables":["o_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_agent2"],"target_variables":["x_next2"],"connection_type":"directed"},{"annotation":null,"source_variables":["C_agent2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["\u03c02"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c02"],"target_variables":["u2"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent2"],"target_variables":["u2"],"connection_type":"directed"},{"annotation":null,"source_variables":["u1"],"target_variables":["s_joint"],"connection_type":"directed"},{"annotation":null,"source_variables":["u2"],"target_variables":["s_joint"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_joint"],"target_variables":["o_joint"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o_agent1"],"target_variables":["s_joint"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o_agent2"],"target_variables":["s_joint"],"connection_type":"undirected"}],"parameters":[{"name":"A_agent1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"A_agent2","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"C_agent1","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"C_agent2","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"D_agent1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"D_agent2","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"B_agent1","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"},{"name":"B_agent2","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"},{"name":"num_hidden_states","value":16,"param_type":"constant"},{"name":"num_obs","value":16,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_agents","value":2,"param_type":"constant"},{"name":"num_hidden_states_per_agent","value":4,"param_type":"constant"},{"name":"num_obs_per_agent","value":4,"param_type":"constant"},{"name":"num_actions_per_agent","value":3,"param_type":"constant"},{"name":"num_timesteps","value":20,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":20,"step_size":null},"ontology_mappings":[{"variable_name":"A_agent1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_agent1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_agent1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_agent1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_agent1","ontology_term":"Agent1HiddenState","description":null},{"variable_name":"x_next1","ontology_term":"Agent1NextHiddenState","description":null},{"variable_name":"o_agent1","ontology_term":"Agent1Observation","description":null},{"variable_name":"\u03c01","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A_agent2","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_agent2","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_agent2","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_agent2","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_agent2","ontology_term":"Agent2HiddenState","description":null},{"variable_name":"x_next2","ontology_term":"Agent2NextHiddenState","description":null},{"variable_name":"o_agent2","ontology_term":"Agent2Observation","description":null},{"variable_name":"\u03c02","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"s_joint","ontology_term":"JointState","description":null},{"variable_name":"o_joint","ontology_term":"JointObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
