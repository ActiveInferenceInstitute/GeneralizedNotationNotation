module ClassicActiveInferencePOMDPAgentv1 where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A = A Double
data B = B Double
data C = C Double
data D = D Double
data E = E Double

-- Connections as Functions
ATos :: A -> s
ATos x = undefined  -- TODO: implement connection
BTos_prime :: B -> s_prime
BTos_prime x = undefined  -- TODO: implement connection
CToo :: C -> o
CToo x = undefined  -- TODO: implement connection
DTos :: D -> s
DTos x = undefined  -- TODO: implement connection
EToπ :: E -> π
EToπ x = undefined  -- TODO: implement connection
sToo :: s -> o
sToo x = undefined  -- TODO: implement connection
πTou :: π -> u
πTou x = undefined  -- TODO: implement connection
uTos_prime :: u -> s_prime
uTos_prime x = undefined  -- TODO: implement connection

-- MODEL_DATA: {"model_name":"Classic Active Inference POMDP Agent v1","annotation":"","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,3]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[3]}],"connections":[{"source_variables":["A"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["o"],"connection_type":"directed"},{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["o"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[],"param_type":"constant"},{"name":"B","value":[],"param_type":"constant"},{"name":"C","value":[],"param_type":"constant"},{"name":"D","value":[],"param_type":"constant"},{"name":"E","value":[],"param_type":"constant"}],"equations":[],"time_specification":null,"ontology_mappings":[]}
