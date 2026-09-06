package gnn.categorical

import cats._
import cats.implicits._
import cats.arrow.Category

object HierarchicalActiveInferencePOMDPModel {

  // State Space
  type A_level1 = Any
  type A_level2 = Any
  type B_level1 = Any
  type B_level2 = Any
  type C_level1 = Any
  type C_level2 = Any
  type D_level1 = Any
  type D_level2 = Any
  type G1 = Any
  type G2 = Any
  type o_level1 = Any
  type o_level2 = Any
  type s_level1 = Any
  type s_level2 = Any
  type t1 = Any
  type t2 = Any
  type u_level1 = Any
  type x_next1 = Any
  type π1 = Any

  // Morphisms
  val A_level1Too_level1: A_level1 => o_level1 = identity
  val A_level2ToD_level1: A_level2 => D_level1 = identity
  val B_level1Tou_level1: B_level1 => u_level1 = identity
  val C_level1ToG1: C_level1 => G1 = identity
  val C_level2ToG2: C_level2 => G2 = identity
  val D_level1Tos_level1: D_level1 => s_level1 = identity
  val D_level2Tos_level2: D_level2 => s_level2 = identity
  val G1Toπ1: G1 => π1 = identity
  val G2Tos_level2: G2 => s_level2 = identity
  val s_level1ToA_level1: s_level1 => A_level1 = identity
  val s_level1Too_level2: s_level1 => o_level2 = identity
  val s_level1Tox_next1: s_level1 => x_next1 = identity
  val s_level2ToA_level2: s_level2 => A_level2 = identity
  val s_level2ToB_level2: s_level2 => B_level2 = identity
  val u_level1Tox_next1: u_level1 => x_next1 = identity
  val π1Tou_level1: π1 => u_level1 = identity

}
// MODEL_DATA: {"model_name":"Hierarchical Active Inference POMDP","annotation":"A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1","variables":[{"name":"A_level1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B_level1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_level1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s_level1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"x_next1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o_level1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u_level1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_level2","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"B_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"C_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"D_level2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s_level2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_level2","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t1","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"t2","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_level1"],"target_variables":["s_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["A_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["A_level1"],"target_variables":["o_level1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"annotation":null,"source_variables":["\u03c01"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_level1"],"target_variables":["u_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_level1"],"target_variables":["x_next1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level1"],"target_variables":["o_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_level2"],"target_variables":["s_level2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["A_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_level2"],"target_variables":["D_level1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_level2"],"target_variables":["B_level2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_level2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["s_level2"],"connection_type":"directed"}],"parameters":[{"name":"A_level1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"B_level1","value":[[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]],[[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C_level1","value":[[0.1,0.1,0.1,1.0]],"param_type":"constant"},{"name":"D_level1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A_level2","value":[[0.9,0.1],[0.1,0.9],[0.5,0.5],[0.5,0.5]],"param_type":"constant"},{"name":"B_level2","value":[[[0.9,0.1],[0.1,0.9]]],"param_type":"constant"},{"name":"C_level2","value":[[0.0,0.5,0.0,0.5]],"param_type":"constant"},{"name":"D_level2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":8,"param_type":"constant"},{"name":"num_obs","value":16,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_timesteps","value":20,"param_type":"constant"},{"name":"num_hidden_states_l1","value":4,"param_type":"constant"},{"name":"num_obs_l1","value":4,"param_type":"constant"},{"name":"num_actions_l1","value":3,"param_type":"constant"},{"name":"num_context_states_l2","value":2,"param_type":"constant"},{"name":"timescale_ratio","value":5,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A_level1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_level1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C_level1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D_level1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s_level1","ontology_term":"HiddenState","description":null},{"variable_name":"o_level1","ontology_term":"Observation","description":null},{"variable_name":"\u03c01","ontology_term":"PolicyVector","description":null},{"variable_name":"u_level1","ontology_term":"Action","description":null},{"variable_name":"G1","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"A_level2","ontology_term":"HigherLevelLikelihoodMatrix","description":null},{"variable_name":"B_level2","ontology_term":"ContextTransitionMatrix","description":null},{"variable_name":"s_level2","ontology_term":"ContextualHiddenState","description":null},{"variable_name":"o_level2","ontology_term":"HigherLevelObservation","description":null},{"variable_name":"G2","ontology_term":"HigherLevelExpectedFreeEnergy","description":null}]}
