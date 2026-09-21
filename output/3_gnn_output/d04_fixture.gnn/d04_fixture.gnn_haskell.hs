module NESTD04Synthetic2StateLGSSM where

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

-- MODEL_DATA: {"model_name":"NEST D04 Synthetic 2-State LGSSM","annotation":"Passive linear-Gaussian state-space model (no control input) generating the\nsummary indices of the D02 MESSAGEix wrapper blanket over T coupling iterations.\n- Hidden state x = (decarb_rate_dev, demand_pressure), dimensionless deviations.\n- Observation y = (emissions_index, price_index, objective_index, demand_index): relative indices of total\n  emissions e, mean commodity price p, objective J and mean demand d against\n  fixed reference values (see the fixture JSON, `blanket.references`).\n- cap (emissions cap) is a declared exogenous schedule in the fixture, not a\n  variable of this model.\nDeliverable D04 (AII); pattern for D07 (3\u20135 states) and D14 (GNN \u2192 RxInfer render).","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[4,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[4,4]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[0.85,-0.1],[0.05,0.9]],"param_type":"constant"},{"name":"H","value":[[-0.8,0.5],[0.6,0.4],[0.3,0.7],[0.0,1.0]],"param_type":"constant"},{"name":"Q","value":[[0.004,0.0005],[0.0005,0.006]],"param_type":"constant"},{"name":"R","value":[[0.0025,0.0,0.0,0.0],[0.0,0.0036,0.0,0.0],[0.0,0.0,0.0016,0.0],[0.0,0.0,0.0,0.0009]],"param_type":"constant"},{"name":"prior_mean","value":[[0.05,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[0.02,0.0],[0.0,0.02]],"param_type":"constant"},{"name":"num_timesteps","value":24,"param_type":"constant"},{"name":"random_seed","value":20260910,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":4,"param_type":"constant"},{"name":"num_regions","value":1,"param_type":"constant"},{"name":"num_commodities","value":2,"param_type":"constant"},{"name":"num_time_slices","value":12,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":24,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
