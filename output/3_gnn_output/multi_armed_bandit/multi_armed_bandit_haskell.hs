module MultiArmedBanditAgent where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A = A Double
data B = B Double
data C = C Double
data D = D Double
data G = G Double
data o = o Int
data s = s Double
data s_prime = s_prime Double
data t = t Int
data u = u Int
data π = π Double

-- Connections as Functions
DTos :: D -> s
DTos x = undefined  -- TODO: implement connection
sToA :: s -> A
sToA x = undefined  -- TODO: implement connection
AToo :: A -> o
AToo x = undefined  -- TODO: implement connection
sTos_prime :: s -> s_prime
sTos_prime x = undefined  -- TODO: implement connection
sToB :: s -> B
sToB x = undefined  -- TODO: implement connection
CToG :: C -> G
CToG x = undefined  -- TODO: implement connection
GToπ :: G -> π
GToπ x = undefined  -- TODO: implement connection
πTou :: π -> u
πTou x = undefined  -- TODO: implement connection
BTou :: B -> u
BTou x = undefined  -- TODO: implement connection
uTos_prime :: u -> s_prime
uTos_prime x = undefined  -- TODO: implement connection

-- MODEL_DATA: {"model_name":"Multi Armed Bandit Agent","annotation":"This model describes a 3-armed bandit as a degenerate POMDP:\n\n- 3 hidden states representing the \"reward context\" (which arm is currently best).\n- 3 observations representing reward signals (no-reward, small-reward, big-reward).\n- 3 actions: pull arm 0, pull arm 1, or pull arm 2.\n- Context switches slowly (sticky transitions), testing exploration vs exploitation.\n- The agent prefers big-reward observations (observation 2).\n- Tests the bandit structure: meaningful actions despite nearly-static state dynamics.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,3]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.1,0.5,0.5],[0.3,0.4,0.3],[0.6,0.1,0.2]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]],"param_type":"constant"},{"name":"C","value":[[0.0,1.0,3.0]],"param_type":"constant"},{"name":"D","value":[[0.33333,0.33333,0.33333]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
