package gnn.categorical

import cats._
import cats.implicits._
import cats.arrow.Category

object ContinuousStateNavigationAgentModel {

  // State Space
  type A_Σ = Any
  type A_μ = Any
  type B_f = Any
  type B_u = Any
  type C_Σ = Any
  type C_μ = Any
  type F = Any
  type G = Any
  type o = Any
  type t = Any
  type u = Any
  type Π_o = Any
  type Π_x = Any
  type Σ = Any
  type ε_o = Any
  type ε_x = Any
  type μ = Any
  type μ_ddot = Any
  type μ_dot = Any
  type μ_prime = Any

  // Morphisms
  val A_ΣToε_o: A_Σ => ε_o = identity
  val A_μToo: A_μ => o = identity
  val B_fToμ_prime: B_f => μ_prime = identity
  val B_uToμ_prime: B_u => μ_prime = identity
  val C_ΣToG: C_Σ => G = identity
  val C_μToG: C_μ => G = identity
  val GTou: G => u = identity
  val oToε_o: o => ε_o = identity
  val uToB_u: u => B_u = identity
  val Π_oToF: Π_o => F = identity
  val Π_xToF: Π_x => F = identity
  val ΣToΠ_o: Σ => Π_o = identity
  val ε_oToF: ε_o => F = identity
  val ε_xToF: ε_x => F = identity
  val μToA_μ: μ => A_μ = identity
  val μToB_f: μ => B_f = identity
  val μToΣ: μ => Σ = identity
  val μ_primeToG: μ_prime => G = identity

}
// MODEL_DATA: {"model_name":"Continuous State Navigation Agent","annotation":"A continuous-state Active Inference agent navigating a 2D environment:\n- Hidden states: 2D position (x, y) as Gaussian belief\n- Observations: noisy position measurements with Gaussian noise\n- Actions: 2D velocity commands (dx, dy)\n- Uses Laplace approximation for Gaussian belief updating\n- Generalized coordinates of motion for smooth trajectories","variables":[{"name":"\u03bc","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03a3","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"\u03bc_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03bc_dot","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03bc_ddot","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"A_\u03bc","var_type":"action","data_type":"float","dimensions":[2,2]},{"name":"A_\u03a3","var_type":"action","data_type":"float","dimensions":[2,2]},{"name":"o","var_type":"observation","data_type":"float","dimensions":[2,1]},{"name":"B_f","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"B_u","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"C_\u03bc","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"C_\u03a3","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"u","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"\u03b5_o","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03b5_x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"\u03a0_o","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"\u03a0_x","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["\u03bc"],"target_variables":["A_\u03bc"],"connection_type":"undirected"},{"source_variables":["A_\u03bc"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["A_\u03a3"],"target_variables":["\u03b5_o"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["\u03b5_o"],"connection_type":"undirected"},{"source_variables":["\u03b5_o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03a0_o"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03bc"],"target_variables":["B_f"],"connection_type":"undirected"},{"source_variables":["B_f"],"target_variables":["\u03bc_prime"],"connection_type":"undirected"},{"source_variables":["B_u"],"target_variables":["\u03bc_prime"],"connection_type":"undirected"},{"source_variables":["u"],"target_variables":["B_u"],"connection_type":"undirected"},{"source_variables":["\u03b5_x"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["\u03a0_x"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["C_\u03bc"],"target_variables":["G"],"connection_type":"undirected"},{"source_variables":["C_\u03a3"],"target_variables":["G"],"connection_type":"undirected"},{"source_variables":["\u03bc_prime"],"target_variables":["G"],"connection_type":"undirected"},{"source_variables":["G"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["\u03bc"],"target_variables":["\u03a3"],"connection_type":"undirected"},{"source_variables":["\u03a3"],"target_variables":["\u03a0_o"],"connection_type":"undirected"}],"parameters":[{"name":"A_\u03bc","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"A_\u03a3","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"B_f","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"B_u","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"C_\u03bc","value":[[1.0],[1.0]],"param_type":"constant"},{"name":"C_\u03a3","value":[[0.05,0.0],[0.0,0.05]],"param_type":"constant"},{"name":"\u03bc","value":[[0.0],[0.0]],"param_type":"constant"},{"name":"\u03a3","value":[[0.5,0.0],[0.0,0.5]],"param_type":"constant"},{"name":"\u03a0_o","value":[[10.0,0.0],[0.0,10.0]],"param_type":"constant"},{"name":"\u03a0_x","value":[[20.0,0.0],[0.0,20.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"10.0","step_size":null},"ontology_mappings":[{"variable_name":"\u03bc","ontology_term":"BeliefMean","description":null},{"variable_name":"\u03a3","ontology_term":"BeliefCovariance","description":null},{"variable_name":"A_\u03bc","ontology_term":"ObservationMeanMapping","description":null},{"variable_name":"A_\u03a3","ontology_term":"ObservationNoise","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"B_f","ontology_term":"DynamicsMatrix","description":null},{"variable_name":"B_u","ontology_term":"ActionEffectMatrix","description":null},{"variable_name":"C_\u03bc","ontology_term":"PreferenceMean","description":null},{"variable_name":"C_\u03a3","ontology_term":"PreferenceCovariance","description":null},{"variable_name":"u","ontology_term":"ContinuousAction","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"\u03b5_o","ontology_term":"SensoryPredictionError","description":null},{"variable_name":"\u03b5_x","ontology_term":"DynamicPredictionError","description":null},{"variable_name":"\u03a0_o","ontology_term":"SensoryPrecision","description":null},{"variable_name":"\u03a0_x","ontology_term":"DynamicPrecision","description":null},{"variable_name":"t","ontology_term":"ContinuousTime","description":null}]}
