-- GNN Model in Lean 4
-- Model: Continuous State Navigation Agent
-- A continuous-state Active Inference agent navigating a 2D environment:
- Hidden states: 2D position (x, y) as Gaussian belief
- Observations: noisy position measurements with Gaussian noise
- Actions: 2D velocity commands (dx, dy)
- Uses Laplace approximation for Gaussian belief updating
- Generalized coordinates of motion for smooth trajectories

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

namespace ContinuousStateNavigationAgent

-- Variables
variable (A_Σ : ℝ)
variable (A_μ : ℝ)
variable (B_f : ℝ)
variable (B_u : ℝ)
variable (C_Σ : ℝ)
variable (C_μ : ℝ)
variable (F : ℝ)
variable (G : ℝ)
variable (o : ℝ)
variable (t : ℝ)
variable (u : ℝ)
variable (Π_o : ℝ)
variable (Π_x : ℝ)
variable (Σ : ℝ)
variable (ε_o : ℝ)
variable (ε_x : ℝ)
variable (μ : ℝ)
variable (μ_ddot : ℝ)
variable (μ_dot : ℝ)
variable (μ_prime : ℝ)

structure ContinuousStateNavigationAgentModel where
  A_Σ : ℝ
  A_μ : ℝ
  B_f : ℝ
  B_u : ℝ
  C_Σ : ℝ
  C_μ : ℝ
  F : ℝ
  G : ℝ
  o : ℝ
  t : ℝ
  u : ℝ
  Π_o : ℝ
  Π_x : ℝ
  Σ : ℝ
  ε_o : ℝ
  ε_x : ℝ
  μ : ℝ
  μ_ddot : ℝ
  μ_dot : ℝ
  μ_prime : ℝ

end ContinuousStateNavigationAgent
-- MODEL_DATA: {"model_name":"Continuous State Navigation Agent","annotation":"A continuous-state Active Inference agent navigating a 2D environment:\n- Hidden states: 2D position (x, y) as Gaussian belief\n- Observations: noisy position measurements with Gaussian noise\n- Actions: 2D velocity commands (dx, dy)\n- Uses Laplace approximation for Gaussian belief updating\n- Generalized coordinates of motion for smooth trajectories","variables":[{"name":"\u03bc","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03a3","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"\u03bc_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03bc_dot","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03bc_ddot","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"A_\u03bc","var_type":"action","data_type":"float","dimensions":[2,2]},{"name":"A_\u03a3","var_type":"action","data_type":"float","dimensions":[2,2]},{"name":"o","var_type":"observation","data_type":"float","dimensions":[2,1]},{"name":"B_f","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"B_u","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"C_\u03bc","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"C_\u03a3","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"u","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"\u03b5_o","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03b5_x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03a0_o","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"\u03a0_x","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["\u03bc"],"target_variables":["A_\u03bc"],"connection_type":"undirected"},{"source_variables":["A_\u03bc"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["A_\u03a3"],"target_variables":["\u03b5_o"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["\u03b5_o"],"connection_type":"undirected"},{"source_variables":["\u03b5_o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03a0_o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03bc"],"target_variables":["B_f"],"connection_type":"undirected"},{"source_variables":["B_f"],"target_variables":["\u03bc_prime"],"connection_type":"undirected"},{"source_variables":["B_u"],"target_variables":["\u03bc_prime"],"connection_type":"undirected"},{"source_variables":["u"],"target_variables":["B_u"],"connection_type":"undirected"},{"source_variables":["\u03b5_x"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03a0_x"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["C_\u03bc"],"target_variables":["G"],"connection_type":"undirected"},{"source_variables":["C_\u03a3"],"target_variables":["G"],"connection_type":"undirected"},{"source_variables":["\u03bc_prime"],"target_variables":["G"],"connection_type":"undirected"},{"source_variables":["G"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["\u03bc"],"target_variables":["\u03a3"],"connection_type":"undirected"},{"source_variables":["\u03a3"],"target_variables":["\u03a0_o"],"connection_type":"undirected"}],"parameters":[{"name":"A_\u03bc","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"A_\u03a3","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"B_f","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"B_u","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"C_\u03bc","value":[[1.0],[1.0]],"param_type":"constant"},{"name":"C_\u03a3","value":[[0.05,0.0],[0.0,0.05]],"param_type":"constant"},{"name":"\u03bc","value":[[0.0],[0.0]],"param_type":"constant"},{"name":"\u03a3","value":[[0.5,0.0],[0.0,0.5]],"param_type":"constant"},{"name":"\u03a0_o","value":[[10.0,0.0],[0.0,10.0]],"param_type":"constant"},{"name":"\u03a0_x","value":[[20.0,0.0],[0.0,20.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"10.0","step_size":null},"ontology_mappings":[{"variable_name":"\u03bc","ontology_term":"BeliefMean","description":null},{"variable_name":"\u03a3","ontology_term":"BeliefCovariance","description":null},{"variable_name":"A_\u03bc","ontology_term":"ObservationMeanMapping","description":null},{"variable_name":"A_\u03a3","ontology_term":"ObservationNoise","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"B_f","ontology_term":"DynamicsMatrix","description":null},{"variable_name":"B_u","ontology_term":"ActionEffectMatrix","description":null},{"variable_name":"C_\u03bc","ontology_term":"PreferenceMean","description":null},{"variable_name":"C_\u03a3","ontology_term":"PreferenceCovariance","description":null},{"variable_name":"u","ontology_term":"ContinuousAction","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"\u03b5_o","ontology_term":"SensoryPredictionError","description":null},{"variable_name":"\u03b5_x","ontology_term":"DynamicPredictionError","description":null},{"variable_name":"\u03a0_o","ontology_term":"SensoryPrecision","description":null},{"variable_name":"\u03a0_x","ontology_term":"DynamicPrecision","description":null},{"variable_name":"t","ontology_term":"ContinuousTime","description":null}]}
