module FactorGraphActiveInferenceModel where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A_prop = A_prop Double
data A_vis = A_vis Double
data B_pos = B_pos Double
data B_vel = B_vel Double
data C_prop = C_prop Double
data C_vis = C_vis Double
data D_pos = D_pos Double
data D_vel = D_vel Double
data F = F Double
data G = G Double
data m_pos_to_vis = m_pos_to_vis Double
data m_prop_to_vel = m_prop_to_vel Double
data m_vel_to_prop = m_vel_to_prop Double
data m_vis_to_pos = m_vis_to_pos Double
data o_prop = o_prop Double
data o_vis = o_vis Int
data s_pos = s_pos Double
data s_vel = s_vel Double
data t = t Int
data u = u Int
data π = π Double

-- Connections as Functions
D_posTos_pos :: D_pos -> s_pos
D_posTos_pos x = undefined
D_velTos_vel :: D_vel -> s_vel
D_velTos_vel x = undefined
s_posToA_vis :: s_pos -> A_vis
s_posToA_vis x = undefined
A_visToo_vis :: A_vis -> o_vis
A_visToo_vis x = undefined
s_velToA_prop :: s_vel -> A_prop
s_velToA_prop x = undefined
A_propToo_prop :: A_prop -> o_prop
A_propToo_prop x = undefined
s_posToB_pos :: s_pos -> B_pos
s_posToB_pos x = undefined
s_velToB_vel :: s_vel -> B_vel
s_velToB_vel x = undefined
B_posTos_pos :: B_pos -> s_pos
B_posTos_pos x = undefined
B_velTos_vel :: B_vel -> s_vel
B_velTos_vel x = undefined
s_posTom_pos_to_vis :: s_pos -> m_pos_to_vis
s_posTom_pos_to_vis x = undefined
m_pos_to_visToA_vis :: m_pos_to_vis -> A_vis
m_pos_to_visToA_vis x = undefined
o_visTom_vis_to_pos :: o_vis -> m_vis_to_pos
o_visTom_vis_to_pos x = undefined
m_vis_to_posTos_pos :: m_vis_to_pos -> s_pos
m_vis_to_posTos_pos x = undefined
s_velTom_vel_to_prop :: s_vel -> m_vel_to_prop
s_velTom_vel_to_prop x = undefined
m_vel_to_propToA_prop :: m_vel_to_prop -> A_prop
m_vel_to_propToA_prop x = undefined
o_propTom_prop_to_vel :: o_prop -> m_prop_to_vel
o_propTom_prop_to_vel x = undefined
m_prop_to_velTos_vel :: m_prop_to_vel -> s_vel
m_prop_to_velTos_vel x = undefined
C_visToG :: C_vis -> G
C_visToG x = undefined
C_propToG :: C_prop -> G
C_propToG x = undefined
GToπ :: G -> π
GToπ x = undefined
πTou :: π -> u
πTou x = undefined
B_posTou :: B_pos -> u
B_posTou x = undefined
s_posToF :: s_pos -> F
s_posToF x = undefined
s_velToF :: s_vel -> F
s_velToF x = undefined
o_visToF :: o_vis -> F
o_visToF x = undefined
o_propToF :: o_prop -> F
o_propToF x = undefined

-- MODEL_DATA: {"model_name":"Factor Graph Active Inference Model","annotation":"A factor graph decomposition of an Active Inference generative model with:\n- Two independent observation modalities (visual and proprioceptive)\n- Two independent hidden state factors (position and velocity)\n- Factored joint distribution: P(o,s) = P(o_vis|s_pos) * P(o_prop|s_vel) * P(s_pos|s_vel) * P(s_vel)\n- Variable nodes: observation and state variables\n- Factor nodes: likelihood and transition factors\n- Enables modality-specific processing and efficient belief propagation","variables":[{"name":"o_vis","var_type":"observation","data_type":"integer","dimensions":[6,1]},{"name":"A_vis","var_type":"action","data_type":"float","dimensions":[6,3]},{"name":"o_prop","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"A_prop","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"s_pos","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"B_pos","var_type":"hidden_state","data_type":"float","dimensions":[3,3,2]},{"name":"s_vel","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"B_vel","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"D_pos","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_vel","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"C_vis","var_type":"hidden_state","data_type":"float","dimensions":[6]},{"name":"C_prop","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"m_pos_to_vis","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"m_vel_to_prop","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"m_vis_to_pos","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"m_prop_to_vel","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D_pos"],"target_variables":["s_pos"],"connection_type":"directed"},{"source_variables":["D_vel"],"target_variables":["s_vel"],"connection_type":"directed"},{"source_variables":["s_pos"],"target_variables":["A_vis"],"connection_type":"undirected"},{"source_variables":["A_vis"],"target_variables":["o_vis"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["A_prop"],"connection_type":"undirected"},{"source_variables":["A_prop"],"target_variables":["o_prop"],"connection_type":"undirected"},{"source_variables":["s_pos"],"target_variables":["B_pos"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["B_vel"],"connection_type":"undirected"},{"source_variables":["B_pos"],"target_variables":["s_pos"],"connection_type":"directed"},{"source_variables":["B_vel"],"target_variables":["s_vel"],"connection_type":"directed"},{"source_variables":["s_pos"],"target_variables":["m_pos_to_vis"],"connection_type":"undirected"},{"source_variables":["m_pos_to_vis"],"target_variables":["A_vis"],"connection_type":"undirected"},{"source_variables":["o_vis"],"target_variables":["m_vis_to_pos"],"connection_type":"undirected"},{"source_variables":["m_vis_to_pos"],"target_variables":["s_pos"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["m_vel_to_prop"],"connection_type":"undirected"},{"source_variables":["m_vel_to_prop"],"target_variables":["A_prop"],"connection_type":"undirected"},{"source_variables":["o_prop"],"target_variables":["m_prop_to_vel"],"connection_type":"undirected"},{"source_variables":["m_prop_to_vel"],"target_variables":["s_vel"],"connection_type":"undirected"},{"source_variables":["C_vis"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["C_prop"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B_pos"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["s_pos"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["s_vel"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_vis"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_prop"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A_vis","value":[[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.05,0.45,0.5],[0.45,0.05,0.5],[0.5,0.5,0.0]],"param_type":"constant"},{"name":"A_prop","value":[[0.9,0.1],[0.1,0.9],[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"B_pos","value":[[[0.9,0.1,0.0],[0.0,0.9,0.1],[0.1,0.0,0.9]],[[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]]],"param_type":"constant"},{"name":"B_vel","value":[[[0.8,0.2],[0.2,0.8]]],"param_type":"constant"},{"name":"D_pos","value":[[0.333,0.333,0.333]],"param_type":"constant"},{"name":"D_vel","value":[[0.5,0.5]],"param_type":"constant"},{"name":"C_vis","value":[[2.0,-0.5,-0.5,-0.5,-0.5,-0.5]],"param_type":"constant"},{"name":"C_prop","value":[[1.0,-1.0,0.0,0.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A_vis","ontology_term":"VisualLikelihoodMatrix","description":null},{"variable_name":"A_prop","ontology_term":"ProprioceptiveLikelihoodMatrix","description":null},{"variable_name":"B_pos","ontology_term":"PositionTransitionMatrix","description":null},{"variable_name":"B_vel","ontology_term":"VelocityTransitionMatrix","description":null},{"variable_name":"D_pos","ontology_term":"PositionPrior","description":null},{"variable_name":"D_vel","ontology_term":"VelocityPrior","description":null},{"variable_name":"C_vis","ontology_term":"VisualPreferenceVector","description":null},{"variable_name":"C_prop","ontology_term":"ProprioceptivePreferenceVector","description":null},{"variable_name":"s_pos","ontology_term":"PositionHiddenState","description":null},{"variable_name":"s_vel","ontology_term":"VelocityHiddenState","description":null},{"variable_name":"o_vis","ontology_term":"VisualObservation","description":null},{"variable_name":"o_prop","ontology_term":"ProprioceptiveObservation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
