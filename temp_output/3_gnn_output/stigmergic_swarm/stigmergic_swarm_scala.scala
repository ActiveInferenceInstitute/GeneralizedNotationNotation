package gnn.categorical

import cats._
import cats.implicits._
import cats.arrow.Category

object StigmergicSwarmActiveInferenceModel {

  // State Space
  type A1 = Any
  type A2 = Any
  type A3 = Any
  type B1 = Any
  type B2 = Any
  type B3 = Any
  type C1 = Any
  type C2 = Any
  type C3 = Any
  type D1 = Any
  type D2 = Any
  type D3 = Any
  type G1 = Any
  type G2 = Any
  type G3 = Any
  type env_signal = Any
  type o1 = Any
  type o2 = Any
  type o3 = Any
  type pi1 = Any
  type pi2 = Any
  type pi3 = Any
  type s1 = Any
  type s2 = Any
  type s3 = Any
  type signal_decay = Any
  type t = Any
  type u1 = Any
  type u2 = Any
  type u3 = Any

  // Morphisms
  val A1Too1: A1 => o1 = identity
  val A2Too2: A2 => o2 = identity
  val A3Too3: A3 => o3 = identity
  val B1Tou1: B1 => u1 = identity
  val B2Tou2: B2 => u2 = identity
  val B3Tou3: B3 => u3 = identity
  val C1ToG1: C1 => G1 = identity
  val C2ToG2: C2 => G2 = identity
  val C3ToG3: C3 => G3 = identity
  val D1Tos1: D1 => s1 = identity
  val D2Tos2: D2 => s2 = identity
  val D3Tos3: D3 => s3 = identity
  val G1Topi1: G1 => pi1 = identity
  val G2Topi2: G2 => pi2 = identity
  val G3Topi3: G3 => pi3 = identity
  val env_signalToA1: env_signal => A1 = identity
  val env_signalToA2: env_signal => A2 = identity
  val env_signalToA3: env_signal => A3 = identity
  val pi1Tou1: pi1 => u1 = identity
  val pi2Tou2: pi2 => u2 = identity
  val pi3Tou3: pi3 => u3 = identity
  val s1ToA1: s1 => A1 = identity
  val s2ToA2: s2 => A2 = identity
  val s3ToA3: s3 => A3 = identity
  val signal_decayToenv_signal: signal_decay => env_signal = identity
  val u1Toenv_signal: u1 => env_signal = identity
  val u2Toenv_signal: u2 => env_signal = identity
  val u3Toenv_signal: u3 => env_signal = identity

}
// MODEL_DATA: {"model_name":"Stigmergic Swarm Active Inference","annotation":"Three Active Inference agents coordinating via stigmergy (environmental traces):\n\n- No direct communication between agents \u2014 coordination emerges from environment\n- Agents deposit and sense environmental signals (pheromone analogy)\n- Shared 3x3 grid environment with signal intensity at each cell\n- Each agent navigates independently while responding to accumulated signals\n- Signal deposition: actions leave traces that other agents can observe\n- Signal decay: environmental signals decay over time (volatility)\n- Demonstrates emergent collective behavior from individual free energy minimization\n- Models ant colony foraging, distributed robotics, and decentralized coordination","variables":[{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A3","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B3","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C3","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D3","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s3","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o3","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi3","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u3","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G3","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"env_signal","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"signal_decay","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"source_variables":["pi1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["o2"],"connection_type":"undirected"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"source_variables":["pi2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["B2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["D3"],"target_variables":["s3"],"connection_type":"directed"},{"source_variables":["s3"],"target_variables":["A3"],"connection_type":"undirected"},{"source_variables":["A3"],"target_variables":["o3"],"connection_type":"undirected"},{"source_variables":["C3"],"target_variables":["G3"],"connection_type":"directed"},{"source_variables":["G3"],"target_variables":["pi3"],"connection_type":"directed"},{"source_variables":["pi3"],"target_variables":["u3"],"connection_type":"directed"},{"source_variables":["B3"],"target_variables":["u3"],"connection_type":"directed"},{"source_variables":["u1"],"target_variables":["env_signal"],"connection_type":"directed"},{"source_variables":["u2"],"target_variables":["env_signal"],"connection_type":"directed"},{"source_variables":["u3"],"target_variables":["env_signal"],"connection_type":"directed"},{"source_variables":["env_signal"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["env_signal"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["env_signal"],"target_variables":["A3"],"connection_type":"undirected"},{"source_variables":["signal_decay"],"target_variables":["env_signal"],"connection_type":"directed"}],"parameters":[{"name":"A1","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A2","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A3","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"C1","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C2","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C3","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"D1","value":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D2","value":[[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D3","value":[[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],"param_type":"constant"},{"name":"env_signal","value":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"signal_decay","value":[[0.9]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":30,"step_size":null},"ontology_mappings":[{"variable_name":"A1","ontology_term":"Agent1LikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"Agent1TransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"Agent1PreferenceVector","description":null},{"variable_name":"D1","ontology_term":"Agent1PositionPrior","description":null},{"variable_name":"s1","ontology_term":"Agent1PositionState","description":null},{"variable_name":"o1","ontology_term":"Agent1Observation","description":null},{"variable_name":"pi1","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"Agent2LikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"Agent2TransitionMatrix","description":null},{"variable_name":"C2","ontology_term":"Agent2PreferenceVector","description":null},{"variable_name":"D2","ontology_term":"Agent2PositionPrior","description":null},{"variable_name":"s2","ontology_term":"Agent2PositionState","description":null},{"variable_name":"o2","ontology_term":"Agent2Observation","description":null},{"variable_name":"pi2","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"A3","ontology_term":"Agent3LikelihoodMatrix","description":null},{"variable_name":"B3","ontology_term":"Agent3TransitionMatrix","description":null},{"variable_name":"C3","ontology_term":"Agent3PreferenceVector","description":null},{"variable_name":"D3","ontology_term":"Agent3PositionPrior","description":null},{"variable_name":"s3","ontology_term":"Agent3PositionState","description":null},{"variable_name":"o3","ontology_term":"Agent3Observation","description":null},{"variable_name":"pi3","ontology_term":"Agent3PolicyVector","description":null},{"variable_name":"u3","ontology_term":"Agent3Action","description":null},{"variable_name":"G3","ontology_term":"Agent3ExpectedFreeEnergy","description":null},{"variable_name":"env_signal","ontology_term":"EnvironmentalSignal","description":null},{"variable_name":"signal_decay","ontology_term":"SignalDecayRate","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
