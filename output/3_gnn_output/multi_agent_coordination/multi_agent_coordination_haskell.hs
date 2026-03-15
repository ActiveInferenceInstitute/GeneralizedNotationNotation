module MultiAgentCooperativeActiveInference where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A1 = A1 Double
data A2 = A2 Double
data B1 = B1 Double
data B2 = B2 Double
data C1 = C1 Double
data C2 = C2 Double
data D1 = D1 Double
data D2 = D2 Double
data G1 = G1 Double
data G2 = G2 Double
data o1 = o1 Int
data o2 = o2 Int
data o_joint = o_joint Int
data s1 = s1 Double
data s1_prime = s1_prime Double
data s2 = s2 Double
data s2_prime = s2_prime Double
data s_joint = s_joint Double
data t = t Int
data u1 = u1 Int
data u2 = u2 Int
data π1 = π1 Double
data π2 = π2 Double

-- Connections as Functions
D1Tos1 :: D1 -> s1
D1Tos1 x = undefined
s1ToA1 :: s1 -> A1
s1ToA1 x = undefined
A1Too1 :: A1 -> o1
A1Too1 x = undefined
s1Tos1_prime :: s1 -> s1_prime
s1Tos1_prime x = undefined
C1ToG1 :: C1 -> G1
C1ToG1 x = undefined
G1Toπ1 :: G1 -> π1
G1Toπ1 x = undefined
π1Tou1 :: π1 -> u1
π1Tou1 x = undefined
B1Tou1 :: B1 -> u1
B1Tou1 x = undefined
D2Tos2 :: D2 -> s2
D2Tos2 x = undefined
s2ToA2 :: s2 -> A2
s2ToA2 x = undefined
A2Too2 :: A2 -> o2
A2Too2 x = undefined
s2Tos2_prime :: s2 -> s2_prime
s2Tos2_prime x = undefined
C2ToG2 :: C2 -> G2
C2ToG2 x = undefined
G2Toπ2 :: G2 -> π2
G2Toπ2 x = undefined
π2Tou2 :: π2 -> u2
π2Tou2 x = undefined
B2Tou2 :: B2 -> u2
B2Tou2 x = undefined
u1Tos_joint :: u1 -> s_joint
u1Tos_joint x = undefined
u2Tos_joint :: u2 -> s_joint
u2Tos_joint x = undefined
s_jointToo_joint :: s_joint -> o_joint
s_jointToo_joint x = undefined
o1Tos_joint :: o1 -> s_joint
o1Tos_joint x = undefined
o2Tos_joint :: o2 -> s_joint
o2Tos_joint x = undefined

-- MODEL_DATA: {"model_name":"Multi-Agent Cooperative Active Inference","annotation":"Two Active Inference agents cooperating on a joint task:\n\n- Agent 1 and Agent 2 each maintain independent beliefs\n- Shared observation space: agents observe each other's actions\n- Joint task state includes both agents' positions (4x4 = 16 joint states)\n- Cooperative preferences: both agents prefer the same goal configuration\n- Models social cognition and coordination without explicit communication","variables":[{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s1_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s2_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c02","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"s_joint","var_type":"hidden_state","data_type":"float","dimensions":[16,1]},{"name":"o_joint","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["s1"],"target_variables":["s1_prime"],"connection_type":"directed"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"source_variables":["\u03c01"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["o2"],"connection_type":"undirected"},{"source_variables":["s2"],"target_variables":["s2_prime"],"connection_type":"directed"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["\u03c02"],"connection_type":"directed"},{"source_variables":["\u03c02"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["B2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["u1"],"target_variables":["s_joint"],"connection_type":"directed"},{"source_variables":["u2"],"target_variables":["s_joint"],"connection_type":"directed"},{"source_variables":["s_joint"],"target_variables":["o_joint"],"connection_type":"undirected"},{"source_variables":["o1"],"target_variables":["s_joint"],"connection_type":"undirected"},{"source_variables":["o2"],"target_variables":["s_joint"],"connection_type":"undirected"}],"parameters":[{"name":"A1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"A2","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"C1","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"C2","value":[[-1.0,-1.0,-1.0,2.0]],"param_type":"constant"},{"name":"D1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"D2","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"B1","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"},{"name":"B2","value":[[[0.9,0.1,0.0,0.0],[0.0,0.9,0.1,0.0],[0.0,0.0,0.9,0.1],[0.1,0.0,0.0,0.9]],[[0.9,0.0,0.0,0.1],[0.1,0.9,0.0,0.0],[0.0,0.1,0.9,0.0],[0.0,0.0,0.1,0.9]],[[0.8,0.1,0.1,0.0],[0.1,0.8,0.0,0.1],[0.1,0.0,0.8,0.1],[0.0,0.1,0.1,0.8]]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":20,"step_size":null},"ontology_mappings":[{"variable_name":"A1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s1","ontology_term":"Agent1HiddenState","description":null},{"variable_name":"s1_prime","ontology_term":"Agent1NextHiddenState","description":null},{"variable_name":"o1","ontology_term":"Agent1Observation","description":null},{"variable_name":"\u03c01","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C2","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D2","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s2","ontology_term":"Agent2HiddenState","description":null},{"variable_name":"s2_prime","ontology_term":"Agent2NextHiddenState","description":null},{"variable_name":"o2","ontology_term":"Agent2Observation","description":null},{"variable_name":"\u03c02","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"s_joint","ontology_term":"JointState","description":null},{"variable_name":"o_joint","ontology_term":"JointObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
