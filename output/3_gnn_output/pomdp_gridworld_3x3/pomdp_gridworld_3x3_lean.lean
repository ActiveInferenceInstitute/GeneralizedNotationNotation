-- GNN Model in Lean 4
-- Model: POMDP GridWorld 3x3
-- Discrete 3x3 GridWorld POMDP for strict cross-framework validation. The model has one hidden state factor with 9 grid cells, one observation modality with noisy cell observations, and one control factor with 5 boundary-clamped actions: up, down, left, right, and stay.

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

namespace POMDPGridWorld3x3

-- Variables
variable (A : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (D : ℝ)
variable (E : ℝ)
variable (G : ℝ)
variable (o : ℤ)
variable (s : ℝ)
variable (s_prime : ℝ)
variable (t : ℤ)
variable (u : ℤ)
variable (π : ℝ)

structure POMDPGridWorld3x3Model where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  G : ℝ
  o : ℤ
  s : ℝ
  s_prime : ℝ
  t : ℤ
  u : ℤ
  π : ℝ

end POMDPGridWorld3x3
-- MODEL_DATA: {"model_name":"POMDP GridWorld 3x3","annotation":"Discrete 3x3 GridWorld POMDP for strict cross-framework validation. The model has one hidden state factor with 9 grid cells, one observation modality with noisy cell observations, and one control factor with 5 boundary-clamped actions: up, down, left, right, and stay.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[9,9]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[9,9,5]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[9]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[9]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[5]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[9,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[5]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.85,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875],[0.01875,0.85,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875],[0.01875,0.01875,0.85,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875],[0.01875,0.01875,0.01875,0.85,0.01875,0.01875,0.01875,0.01875,0.01875],[0.01875,0.01875,0.01875,0.01875,0.85,0.01875,0.01875,0.01875,0.01875],[0.01875,0.01875,0.01875,0.01875,0.01875,0.85,0.01875,0.01875,0.01875],[0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.85,0.01875,0.01875],[0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.85,0.01875],[0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.01875,0.85]],"param_type":"constant"},{"name":"B","value":[[[1.0,0.0,1.0,0.0,1.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[1.0,0.0,0.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,1.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,1.0,1.0,0.0,1.0],[0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,0.0,1.0],[0.0,0.0,1.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0],[0.0,1.0,0.0,1.0,1.0]]],"param_type":"constant"},{"name":"C","value":[[0.0,0.1,0.3,0.1,0.4,0.8,0.3,0.8,3.0]],"param_type":"constant"},{"name":"D","value":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"E","value":[[0.2,0.2,0.2,0.2,0.2]],"param_type":"constant"},{"name":"num_hidden_states","value":9,"param_type":"constant"},{"name":"num_obs","value":9,"param_type":"constant"},{"name":"num_actions","value":5,"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"},{"name":"random_seed","value":42,"param_type":"constant"},{"name":"b_tensor_order","value":"next_state_previous_state_action","param_type":"constant"},{"name":"grid_rows","value":3,"param_type":"constant"},{"name":"grid_cols","value":3,"param_type":"constant"},{"name":"goal_state","value":8,"param_type":"constant"},{"name":"action_labels","value":"up,down,left,right,stay","param_type":"constant"}],"equations":["Equation(node_type='Equation', source_location=None, metadata={}, id='2b0cae24-4711-4ca2-a451-62c79935c538', label=None, content='State inference uses the observation likelihood and previous predictive belief. Action selection minimizes expected free energy under the shared transition tensor.', format='latex', description=None)"],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"Habit","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
