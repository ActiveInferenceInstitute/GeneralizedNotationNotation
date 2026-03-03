module HiddenMarkovModelBaseline where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A = A Double
data B = B Double
data D = D Double
data F = F Double
data alpha = alpha Double
data beta = beta Double
data o = o Int
data s = s Double
data s_prime = s_prime Double
data t = t Int

-- Connections as Functions
DTos :: D -> s
DTos x = undefined  -- TODO: implement connection
sToA :: s -> A
sToA x = undefined  -- TODO: implement connection
sTos_prime :: s -> s_prime
sTos_prime x = undefined  -- TODO: implement connection
AToo :: A -> o
AToo x = undefined  -- TODO: implement connection
BTos_prime :: B -> s_prime
BTos_prime x = undefined  -- TODO: implement connection
sToB :: s -> B
sToB x = undefined  -- TODO: implement connection
sToF :: s -> F
sToF x = undefined  -- TODO: implement connection
oToF :: o -> F
oToF x = undefined  -- TODO: implement connection
sToalpha :: s -> alpha
sToalpha x = undefined  -- TODO: implement connection
oToalpha :: o -> alpha
oToalpha x = undefined  -- TODO: implement connection
alphaTos_prime :: alpha -> s_prime
alphaTos_prime x = undefined  -- TODO: implement connection
s_primeTobeta :: s_prime -> beta
s_primeTobeta x = undefined  -- TODO: implement connection

-- MODEL_DATA: {"model_name":"Hidden Markov Model Baseline","annotation":"A standard discrete Hidden Markov Model with:\n- 4 hidden states with Markovian dynamics\n- 6 observation symbols\n- Fixed transition and emission matrices\n- No action selection (passive inference only)\n- Suitable for sequence modeling and state estimation tasks","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[6,4]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[4,4]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[4]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[6,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"alpha","var_type":"action","data_type":"float","dimensions":[4,1]},{"name":"beta","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["alpha"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["alpha"],"connection_type":"undirected"},{"source_variables":["alpha"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s_prime"],"target_variables":["beta"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.1,0.1,0.7,0.1],[0.1,0.1,0.1,0.7],[0.1,0.1,0.4,0.4],[0.4,0.4,0.1,0.1]],"param_type":"constant"},{"name":"B","value":[[0.7,0.1,0.1,0.1],[0.1,0.7,0.2,0.1],[0.1,0.1,0.6,0.2],[0.1,0.1,0.1,0.6]],"param_type":"constant"},{"name":"D","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"EmissionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"InitialStateDistribution","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"alpha","ontology_term":"ForwardVariable","description":null},{"variable_name":"beta","ontology_term":"BackwardVariable","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
