module StochasticContinuousDynamicsAgent where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data F = F Double
data G = G Double
data H = H Double
data epsilon_obs = epsilon_obs Double
data epsilon_state = epsilon_state Double
data gamma_obs = gamma_obs Double
data gamma_state = gamma_state Double
data o = o Double
data u = u Double
data x = x Double

-- Connections as Functions
xToF :: x -> F
xToF x = undefined
uToF :: u -> F
uToF x = undefined
epsilon_stateToF :: epsilon_state -> F
epsilon_stateToF x = undefined
FTox :: F -> x
FTox x = undefined
xToH :: x -> H
xToH x = undefined
epsilon_obsToH :: epsilon_obs -> H
epsilon_obsToH x = undefined
HToo :: H -> o
HToo x = undefined
gamma_stateToepsilon_state :: gamma_state -> epsilon_state
gamma_stateToepsilon_state x = undefined
gamma_obsToepsilon_obs :: gamma_obs -> epsilon_obs
gamma_obsToepsilon_obs x = undefined

-- MODEL_DATA: {"model_name":"Stochastic Continuous Dynamics Agent","annotation":"A continuous-state Active Inference agent whose dynamics include\nexplicit process and observation noise. This sample exercises the\ncontinuous-time path of the language: Time=Continuous, state as a\nmulti-dimensional vector, and Equations containing noise terms \u03b5_state\nand \u03b5_obs that downstream renderers must handle.\n\n- 4-dimensional continuous hidden state (e.g., position + velocity in 2D)\n- 2-dimensional continuous observation (noisy position readout)\n- Linear Gaussian dynamics: \u1e8b = Fx + Gu + \u03b5_state\n- Linear Gaussian observation: o = Hx + \u03b5_obs\n- Precision parameters \u03b3_state (process) and \u03b3_obs (observation)\n\nUsed to validate that downstream backends either generate SDE solvers\n(Stan via latent-variable sampling, NumPyro) or raise a clean\n\"unsupported feature\" warning (PyMDP, which is discrete-only).","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o","var_type":"observation","data_type":"float","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[4,4]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[4,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[2,4]},{"name":"gamma_state","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"gamma_obs","var_type":"observation","data_type":"float","dimensions":[1]},{"name":"epsilon_state","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"epsilon_obs","var_type":"observation","data_type":"float","dimensions":[2,1]}],"connections":[{"source_variables":["x","u","epsilon_state"],"target_variables":["F"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["x","epsilon_obs"],"target_variables":["H"],"connection_type":"directed"},{"source_variables":["H"],"target_variables":["o"],"connection_type":"directed"},{"source_variables":["gamma_state"],"target_variables":["epsilon_state"],"connection_type":"undirected"},{"source_variables":["gamma_obs"],"target_variables":["epsilon_obs"],"connection_type":"undirected"}],"parameters":[{"name":"F","value":[[1.0,0.0,0.1,0.0],[0.0,1.0,0.0,0.1],[0.0,0.0,0.95,0.0],[0.0,0.0,0.0,0.95]],"param_type":"constant"},{"name":"G","value":[[0.0,0.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"H","value":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]],"param_type":"constant"},{"name":"gamma_state","value":[[10.0]],"param_type":"constant"},{"name":"gamma_obs","value":[[5.0]],"param_type":"constant"},{"name":"state_dim","value":4,"param_type":"constant"},{"name":"obs_dim","value":2,"param_type":"constant"},{"name":"control_dim","value":2,"param_type":"constant"},{"name":"num_timesteps","value":50,"param_type":"constant"},{"name":"time_horizon","value":5.0,"param_type":"constant"},{"name":"integration_step","value":0.1,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Static","discretization":"ContinuousTime","horizon":"5.0","step_size":null},"ontology_mappings":[{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"o","ontology_term":"ContinuousObservation","description":null},{"variable_name":"u","ontology_term":"ContinuousAction","description":null},{"variable_name":"F","ontology_term":"DriftMatrix","description":null},{"variable_name":"G","ontology_term":"ControlMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"gamma_state","ontology_term":"ProcessNoisePrecision","description":null},{"variable_name":"gamma_obs","ontology_term":"ObservationNoisePrecision","description":null},{"variable_name":"epsilon_state","ontology_term":"ProcessNoise","description":null},{"variable_name":"epsilon_obs","ontology_term":"ObservationNoise","description":null}]}
