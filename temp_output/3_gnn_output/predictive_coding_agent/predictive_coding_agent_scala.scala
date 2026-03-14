package gnn.categorical

import cats._
import cats.implicits._
import cats.arrow.Category

object PredictiveCodingActiveInferenceAgentModel {

  // State Space
  type F = Any
  type F_d = Any
  type F_s = Any
  type Pi_d = Any
  type Pi_s = Any
  type Sigma = Any
  type e_d = Any
  type e_s = Any
  type f_params = Any
  type g_params = Any
  type mu = Any
  type mu_dot = Any
  type mu_star = Any
  type o = Any
  type t = Any
  type u = Any

  // Morphisms
  val FTomu: F => mu = identity
  val FTou: F => u = identity
  val F_dToF: F_d => F = identity
  val F_sToF: F_s => F = identity
  val Pi_dToe_d: Pi_d => e_d = identity
  val Pi_sToe_s: Pi_s => e_s = identity
  val SigmaToPi_s: Sigma => Pi_s = identity
  val e_dToF_d: e_d => F_d = identity
  val e_sToF_s: e_s => F_s = identity
  val f_paramsTomu_dot: f_params => mu_dot = identity
  val g_paramsToo: g_params => o = identity
  val muToSigma: mu => Sigma = identity
  val muToe_s: mu => e_s = identity
  val muTof_params: mu => f_params = identity
  val muTog_params: mu => g_params = identity
  val mu_dotToe_d: mu_dot => e_d = identity
  val mu_starToF: mu_star => F = identity
  val oToe_s: o => e_s = identity

}
// MODEL_DATA: {"model_name":"Predictive Coding Active Inference Agent","annotation":"A continuous-state Active Inference agent implementing predictive coding:\n\n- Two-level predictive hierarchy: sensory predictions and dynamics predictions\n- Prediction errors drive belief updating via gradient descent on free energy\n- Precision-weighted prediction errors enable attentional modulation\n- Sensory level: predicts observations from hidden causes\n- Dynamics level: predicts state evolution from generative dynamics\n- Action minimizes expected free energy by changing sensory input\n- Uses generalized coordinates of motion (position, velocity, acceleration)\n- Demonstrates the core predictive processing framework underlying Active Inference","variables":[{"name":"mu","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"mu_dot","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"Sigma","var_type":"hidden_state","data_type":"float","dimensions":[3,3]},{"name":"e_s","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"e_d","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"g_params","var_type":"hidden_state","data_type":"float","dimensions":[12]},{"name":"f_params","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"Pi_s","var_type":"policy","data_type":"float","dimensions":[4,4]},{"name":"Pi_d","var_type":"policy","data_type":"float","dimensions":[3,3]},{"name":"o","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"u","var_type":"action","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"F_s","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"F_d","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"mu_star","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"t","var_type":"hidden_state","data_type":"float","dimensions":[1]}],"connections":[{"source_variables":["mu"],"target_variables":["g_params"],"connection_type":"undirected"},{"source_variables":["g_params"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["o"],"target_variables":["e_s"],"connection_type":"undirected"},{"source_variables":["mu"],"target_variables":["e_s"],"connection_type":"undirected"},{"source_variables":["Pi_s"],"target_variables":["e_s"],"connection_type":"undirected"},{"source_variables":["e_s"],"target_variables":["F_s"],"connection_type":"undirected"},{"source_variables":["mu"],"target_variables":["f_params"],"connection_type":"undirected"},{"source_variables":["f_params"],"target_variables":["mu_dot"],"connection_type":"undirected"},{"source_variables":["mu_dot"],"target_variables":["e_d"],"connection_type":"undirected"},{"source_variables":["Pi_d"],"target_variables":["e_d"],"connection_type":"undirected"},{"source_variables":["e_d"],"target_variables":["F_d"],"connection_type":"undirected"},{"source_variables":["F_s"],"target_variables":["F"],"connection_type":"directed"},{"source_variables":["F_d"],"target_variables":["F"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["mu"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["mu_star"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["mu"],"target_variables":["Sigma"],"connection_type":"undirected"},{"source_variables":["Sigma"],"target_variables":["Pi_s"],"connection_type":"undirected"}],"parameters":[{"name":"mu","value":[[0.0],[0.0],[0.0]],"param_type":"constant"},{"name":"mu_dot","value":[[0.0],[0.0],[0.0]],"param_type":"constant"},{"name":"Sigma","value":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"Pi_s","value":[[8.0,0.0,0.0,0.0],[0.0,8.0,0.0,0.0],[0.0,0.0,8.0,0.0],[0.0,0.0,0.0,8.0]],"param_type":"constant"},{"name":"Pi_d","value":[[4.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,4.0]],"param_type":"constant"},{"name":"mu_star","value":[[1.0],[1.0],[0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"5.0","step_size":null},"ontology_mappings":[{"variable_name":"mu","ontology_term":"BeliefMean","description":null},{"variable_name":"mu_dot","ontology_term":"BeliefVelocity","description":null},{"variable_name":"Sigma","ontology_term":"BeliefCovariance","description":null},{"variable_name":"e_s","ontology_term":"SensoryPredictionError","description":null},{"variable_name":"e_d","ontology_term":"DynamicPredictionError","description":null},{"variable_name":"g_params","ontology_term":"SensoryMappingParameters","description":null},{"variable_name":"f_params","ontology_term":"DynamicsParameters","description":null},{"variable_name":"Pi_s","ontology_term":"SensoryPrecision","description":null},{"variable_name":"Pi_d","ontology_term":"DynamicPrecision","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"u","ontology_term":"ContinuousAction","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"F_s","ontology_term":"SensoryFreeEnergy","description":null},{"variable_name":"F_d","ontology_term":"DynamicFreeEnergy","description":null},{"variable_name":"mu_star","ontology_term":"PriorExpectation","description":null},{"variable_name":"t","ontology_term":"ContinuousTime","description":null}]}
