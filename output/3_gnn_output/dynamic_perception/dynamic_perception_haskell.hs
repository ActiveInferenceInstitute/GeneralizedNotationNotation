module DynamicPerceptionModel where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A = A Double
data B = B Double
data D = D Double
data F = F Double
data o_t = o_t Int
data s_prime = s_prime Double
data s_t = s_t Double
data t = t Int

-- Connections as Functions
DTos_t :: D -> s_t
DTos_t x = undefined  -- TODO: implement connection
s_tToA :: s_t -> A
s_tToA x = undefined  -- TODO: implement connection
AToo_t :: A -> o_t
AToo_t x = undefined  -- TODO: implement connection
s_tToB :: s_t -> B
s_tToB x = undefined  -- TODO: implement connection
BTos_prime :: B -> s_prime
BTos_prime x = undefined  -- TODO: implement connection
s_tToF :: s_t -> F
s_tToF x = undefined  -- TODO: implement connection
o_tToF :: o_t -> F
o_tToF x = undefined  -- TODO: implement connection

-- MODEL_DATA: {"model_name":"Dynamic Perception Model","annotation":"A dynamic perception model extending the static model with temporal dynamics:\n\n- 2 hidden states evolving over discrete time via transition matrix B\n- 2 observations generated from states via recognition matrix A\n- Prior D constrains the initial hidden state\n- No action selection \u2014 the agent passively observes a changing world\n- Demonstrates belief updating (state inference) across time steps\n- Suitable for tracking hidden sources from noisy observations","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"s_t","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_t","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s_t"],"connection_type":"directed"},{"source_variables":["s_t"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o_t"],"connection_type":"undirected"},{"source_variables":["s_t"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s_t"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_t"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[0.7,0.3],[0.3,0.7]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":10,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s_t","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o_t","ontology_term":"Observation","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
