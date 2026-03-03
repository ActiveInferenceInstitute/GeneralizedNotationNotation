-- GNN Model in Lean 4
-- Model: Dynamic Perception Model
-- A dynamic perception model extending the static model with temporal dynamics:

- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via recognition matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

namespace DynamicPerceptionModel

-- Variables
variable (A : ℝ)
variable (B : ℝ)
variable (D : ℝ)
variable (F : ℝ)
variable (o_t : ℤ)
variable (s_prime : ℝ)
variable (s_t : ℝ)
variable (t : ℤ)

structure DynamicPerceptionModelModel where
  A : ℝ
  B : ℝ
  D : ℝ
  F : ℝ
  o_t : ℤ
  s_prime : ℝ
  s_t : ℝ
  t : ℤ

end DynamicPerceptionModel
-- MODEL_DATA: {"model_name":"Dynamic Perception Model","annotation":"A dynamic perception model extending the static model with temporal dynamics:\n\n- 2 hidden states evolving over discrete time via transition matrix B\n- 2 observations generated from states via recognition matrix A\n- Prior D constrains the initial hidden state\n- No action selection \u2014 the agent passively observes a changing world\n- Demonstrates belief updating (state inference) across time steps\n- Suitable for tracking hidden sources from noisy observations","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"s_t","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_t","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s_t"],"connection_type":"directed"},{"source_variables":["s_t"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o_t"],"connection_type":"undirected"},{"source_variables":["s_t"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s_t"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_t"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[0.7,0.3],[0.3,0.7]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":10,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s_t","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o_t","ontology_term":"Observation","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
