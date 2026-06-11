module FactorizedPosteriorAgent where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A_m0 = A_m0 Double
data A_m1 = A_m1 Double
data B_f0 = B_f0 Double
data B_f1 = B_f1 Double
data C_m0 = C_m0 Double
data C_m1 = C_m1 Double
data D_f0 = D_f0 Double
data D_f1 = D_f1 Double
data o_m0 = o_m0 Int
data o_m1 = o_m1 Int
data s_f0 = s_f0 Double
data s_f1 = s_f1 Double
data u = u Int

-- Connections as Functions
D_f0Tos_f0 :: D_f0 -> s_f0
D_f0Tos_f0 x = undefined
D_f1Tos_f1 :: D_f1 -> s_f1
D_f1Tos_f1 x = undefined
s_f0ToB_f0 :: s_f0 -> B_f0
s_f0ToB_f0 x = undefined
uToB_f0 :: u -> B_f0
uToB_f0 x = undefined
B_f0Tos_f0 :: B_f0 -> s_f0
B_f0Tos_f0 x = undefined
s_f1ToB_f1 :: s_f1 -> B_f1
s_f1ToB_f1 x = undefined
B_f1Tos_f1 :: B_f1 -> s_f1
B_f1Tos_f1 x = undefined
s_f0ToA_m0 :: s_f0 -> A_m0
s_f0ToA_m0 x = undefined
s_f1ToA_m0 :: s_f1 -> A_m0
s_f1ToA_m0 x = undefined
A_m0Too_m0 :: A_m0 -> o_m0
A_m0Too_m0 x = undefined
s_f0ToA_m1 :: s_f0 -> A_m1
s_f0ToA_m1 x = undefined
A_m1Too_m1 :: A_m1 -> o_m1
A_m1Too_m1 x = undefined
C_m0Too_m0 :: C_m0 -> o_m0
C_m0Too_m0 x = undefined
C_m1Too_m1 :: C_m1 -> o_m1
C_m1Too_m1 x = undefined

-- MODEL_DATA: {"model_name":"Factorized Posterior Agent","annotation":"A mean-field factorized POMDP agent. The joint posterior over two\nindependent state factors `s_1` (location) and `s_2` (goal identity) is\napproximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).\nThis is the canonical simplification used in variational inference when\nexact joint posteriors are computationally intractable.\n\n- Two state factors: location (4 states), goal (2 states)\n- Two observation modalities: visual (3 obs), proprioceptive (2 obs)\n- Separate transition matrices B_1 (location \u00d7 action) and B_2 (goal is static)\n- Explicit factorization declared in ## Equations\n- Tests multi-factor / multi-modality handling in the parser","variables":[{"name":"s_f0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_f1","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_m0","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"o_m1","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[3,1]},{"name":"A_m0","var_type":"action","data_type":"float","dimensions":[3,4,2]},{"name":"A_m1","var_type":"action","data_type":"float","dimensions":[2,4]},{"name":"B_f0","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"B_f1","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"D_f0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"D_f1","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"C_m0","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"C_m1","var_type":"hidden_state","data_type":"float","dimensions":[2,1]}],"connections":[{"source_variables":["D_f0"],"target_variables":["s_f0"],"connection_type":"directed"},{"source_variables":["D_f1"],"target_variables":["s_f1"],"connection_type":"directed"},{"source_variables":["s_f0","u"],"target_variables":["B_f0"],"connection_type":"directed"},{"source_variables":["B_f0"],"target_variables":["s_f0"],"connection_type":"directed"},{"source_variables":["s_f1"],"target_variables":["B_f1"],"connection_type":"directed"},{"source_variables":["B_f1"],"target_variables":["s_f1"],"connection_type":"directed"},{"source_variables":["s_f0","s_f1"],"target_variables":["A_m0"],"connection_type":"directed"},{"source_variables":["A_m0"],"target_variables":["o_m0"],"connection_type":"directed"},{"source_variables":["s_f0"],"target_variables":["A_m1"],"connection_type":"directed"},{"source_variables":["A_m1"],"target_variables":["o_m1"],"connection_type":"directed"},{"source_variables":["C_m0"],"target_variables":["o_m0"],"connection_type":"undirected"},{"source_variables":["C_m1"],"target_variables":["o_m1"],"connection_type":"undirected"}],"parameters":[{"name":"A_m0","value":[[[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.2,0.2,0.8,0.8]],[[0.1,0.7,0.1,0.1],[0.7,0.1,0.1,0.1],[0.2,0.2,0.8,0.8]]],"param_type":"constant"},{"name":"A_m1","value":[[0.9,0.1,0.1,0.1],[0.1,0.9,0.9,0.9]],"param_type":"constant"},{"name":"D_f0","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"D_f1","value":[[0.6,0.4]],"param_type":"constant"},{"name":"C_m0","value":[[0.0,0.0,1.0]],"param_type":"constant"},{"name":"C_m1","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states_factor0","value":4,"param_type":"constant"},{"name":"num_hidden_states_factor1","value":2,"param_type":"constant"},{"name":"num_obs_modality0","value":3,"param_type":"constant"},{"name":"num_obs_modality1","value":2,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_factors","value":2,"param_type":"constant"},{"name":"num_modalities","value":2,"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":"DiscreteTime","horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"s_f0","ontology_term":"HiddenStateFactor0","description":null},{"variable_name":"s_f1","ontology_term":"HiddenStateFactor1","description":null},{"variable_name":"o_m0","ontology_term":"ObservationModality0","description":null},{"variable_name":"o_m1","ontology_term":"ObservationModality1","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"A_m0","ontology_term":"LikelihoodMatrixModality0","description":null},{"variable_name":"A_m1","ontology_term":"LikelihoodMatrixModality1","description":null},{"variable_name":"B_f0","ontology_term":"TransitionMatrixFactor0","description":null},{"variable_name":"B_f1","ontology_term":"TransitionMatrixFactor1","description":null},{"variable_name":"D_f0","ontology_term":"PriorFactor0","description":null},{"variable_name":"D_f1","ontology_term":"PriorFactor1","description":null},{"variable_name":"C_m0","ontology_term":"PreferenceModality0","description":null},{"variable_name":"C_m1","ontology_term":"PreferenceModality1","description":null}]}
