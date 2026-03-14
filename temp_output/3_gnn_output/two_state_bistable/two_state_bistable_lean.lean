-- GNN Model in Lean 4
-- Model: Two State Bistable POMDP
-- This model describes a minimal 2-state bistable POMDP:

- 2 hidden states: "left" and "right" in a symmetric bistable potential.
- 2 noisy observations: the agent gets a noisy readout of which side it is on.
- 2 actions: push-left or push-right.
- The agent prefers observation 1 ("right") over observation 0 ("left").
- Tests the absolute smallest POMDP with full active inference structure.

import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic

namespace TwoStateBistablePOMDP

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

structure TwoStateBistablePOMDPModel where
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

end TwoStateBistablePOMDP
-- MODEL_DATA: {"model_name":"Two State Bistable POMDP","annotation":"This model describes a minimal 2-state bistable POMDP:\n\n- 2 hidden states: \"left\" and \"right\" in a symmetric bistable potential.\n- 2 noisy observations: the agent gets a noisy readout of which side it is on.\n- 2 actions: push-left or push-right.\n- The agent prefers observation 1 (\"right\") over observation 0 (\"left\").\n- Tests the absolute smallest POMDP with full active inference structure.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"E","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["E"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.8,0.2],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[[0.8,0.3],[0.2,0.7]],[[0.3,0.8],[0.7,0.2]]],"param_type":"constant"},{"name":"C","value":[[0.0,2.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"},{"name":"E","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"E","ontology_term":"Habit","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
