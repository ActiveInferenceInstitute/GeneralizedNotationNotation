module StochasticContinuousDynamicsAgent where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data F = F Double
data H = H Double
data Q = Q Double
data R = R Double
data prior_cov = prior_cov Double
data prior_mean = prior_mean Double
data t = t Int
data x = x Double
data y = y Double

-- Connections as Functions
prior_meanTox :: prior_mean -> x
prior_meanTox x = undefined
FTox :: F -> x
FTox x = undefined
xToy :: x -> y
xToy x = undefined
HToy :: H -> y
HToy x = undefined
QTox :: Q -> x
QTox x = undefined
RToy :: R -> y
RToy x = undefined

-- MODEL_DATA: {"model_name":"Stochastic Continuous Dynamics Agent","annotation":"A continuous-state Active Inference agent whose dynamics carry explicit process\nand observation noise, rendered as a native linear-Gaussian state-space model\n(LGSSM). The agent runs passively \u2014 it has no control input:\n- Hidden state x = (position, velocity): the Euler-discretized (dt = 0.1) SDE.\n- Observation y: two noisy readouts, both reading the position.\n- Q is the process-noise covariance (inverse process precision); R is the\n  observation-noise covariance (inverse observation precision).","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[1.0,0.1],[0.0,0.9]],"param_type":"constant"},{"name":"H","value":[[1.0,0.0],[1.0,0.0]],"param_type":"constant"},{"name":"Q","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"R","value":[[0.2,0.0],[0.0,0.2]],"param_type":"constant"},{"name":"prior_mean","value":[[0.0,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[0.5,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"},{"name":"dt","value":0.1,"param_type":"constant"},{"name":"random_seed","value":42,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":2,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
