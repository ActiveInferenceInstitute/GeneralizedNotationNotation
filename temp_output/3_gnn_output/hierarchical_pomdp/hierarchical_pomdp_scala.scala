package gnn.categorical

import cats._
import cats.implicits._
import cats.arrow.Category

object HierarchicalActiveInferencePOMDPModel {

  // State Space
  type A1 = Any
  type A2 = Any
  type B1 = Any
  type B2 = Any
  type C1 = Any
  type C2 = Any
  type D1 = Any
  type D2 = Any
  type G1 = Any
  type G2 = Any
  type o1 = Any
  type o2 = Any
  type s1 = Any
  type s1_prime = Any
  type s2 = Any
  type t1 = Any
  type t2 = Any
  type u1 = Any
  type π1 = Any

  // Morphisms
  val A1Too1: A1 => o1 = identity
  val A2ToD1: A2 => D1 = identity
  val B1Tou1: B1 => u1 = identity
  val C1ToG1: C1 => G1 = identity
  val C2ToG2: C2 => G2 = identity
  val D1Tos1: D1 => s1 = identity
  val D2Tos2: D2 => s2 = identity
  val G1Toπ1: G1 => π1 = identity
  val G2Tos2: G2 => s2 = identity
  val s1ToA1: s1 => A1 = identity
  val s1Too2: s1 => o2 = identity
  val s1Tos1_prime: s1 => s1_prime = identity
  val s2ToA2: s2 => A2 = identity
  val s2ToB2: s2 => B2 = identity
  val u1Tos1_prime: u1 => s1_prime = identity
  val π1Tou1: π1 => u1 = identity

}
// MODEL_DATA: {"model_name":"Hierarchical Active Inference POMDP","annotation":"A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1","variables":[{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s1_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"\u03c01","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[4,2]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o2","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t1","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"t2","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["s1"],"target_variables":["s1_prime"],"connection_type":"directed"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["\u03c01"],"connection_type":"directed"},{"source_variables":["\u03c01"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["u1"],"target_variables":["s1_prime"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["o2"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["D1"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["B2"],"connection_type":"undirected"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["s2"],"connection_type":"directed"}],"parameters":[{"name":"A1","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05],[0.05,0.05,0.05,0.85]],"param_type":"constant"},{"name":"B1","value":[[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0]],[[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]],"param_type":"constant"},{"name":"C1","value":[[0.1,0.1,0.1,1.0]],"param_type":"constant"},{"name":"D1","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A2","value":[[0.9,0.1,0.0,0.0],[0.1,0.9,0.0,0.0],[0.0,0.0,0.9,0.1],[0.0,0.0,0.1,0.9]],"param_type":"constant"},{"name":"B2","value":[[[0.9,0.1],[0.1,0.9]]],"param_type":"constant"},{"name":"C2","value":[[0.1,1.0]],"param_type":"constant"},{"name":"D2","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A1","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D1","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"s1","ontology_term":"HiddenState","description":null},{"variable_name":"o1","ontology_term":"Observation","description":null},{"variable_name":"\u03c01","ontology_term":"PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Action","description":null},{"variable_name":"G1","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"HigherLevelLikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"ContextTransitionMatrix","description":null},{"variable_name":"s2","ontology_term":"ContextualHiddenState","description":null},{"variable_name":"o2","ontology_term":"HigherLevelObservation","description":null},{"variable_name":"G2","ontology_term":"HigherLevelExpectedFreeEnergy","description":null}]}
