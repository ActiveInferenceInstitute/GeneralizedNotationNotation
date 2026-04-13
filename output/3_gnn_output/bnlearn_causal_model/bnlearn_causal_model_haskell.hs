module BnlearnCausalModel where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A = A Double
data B = B Double
data a = a Int
data o = o Int
data s = s Double
data s_prev = s_prev Double

-- Connections as Functions
s_prevTos :: s_prev -> s
s_prevTos x = undefined
aTos :: a -> s
aTos x = undefined
sToo :: s -> o
sToo x = undefined

-- MODEL_DATA: {"model_name":"Bnlearn Causal Model","annotation":"A Bayesian Network model mapping Active Inference structure:\n- S: Hidden State\n- A: Action\n- S_prev: Previous State\n- O: Observation","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prev","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"a","var_type":"action","data_type":"integer","dimensions":[2,1]}],"connections":[{"source_variables":["s_prev"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["a"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["o"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B","value":[[[0.7,0.3],[0.3,0.7]],[[0.3,0.7],[0.7,0.3]]],"param_type":"constant"},{"name":"C","value":[[0.0,1.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"ObservationModel","description":null},{"variable_name":"B","ontology_term":"TransitionModel","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prev","ontology_term":"PreviousState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"a","ontology_term":"Action","description":null}]}
