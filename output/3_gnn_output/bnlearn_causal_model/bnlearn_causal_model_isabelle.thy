theory BnlearnCausalModel
imports Main
begin

text \<open>Bnlearn Causal Model\<close>
text \<open>A Bayesian Network model mapping Active Inference structure:
- S: Hidden State
- A: Action
- S_prev: Previous State
- O: Observation\<close>

type_synonym A = "real"
type_synonym B = "real"
type_synonym a = "int"
type_synonym o = "int"
type_synonym s = "real"
type_synonym s_prev = "real"

end
(* MODEL_DATA: {"model_name":"Bnlearn Causal Model","annotation":"A Bayesian Network model mapping Active Inference structure:\n- S: Hidden State\n- A: Action\n- S_prev: Previous State\n- O: Observation","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prev","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"a","var_type":"action","data_type":"integer","dimensions":[2,1]}],"connections":[{"source_variables":["s_prev"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["a"],"target_variables":["s"],"connection_type":"directed"},{"source_variables":["s"],"target_variables":["o"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"ObservationModel","description":null},{"variable_name":"B","ontology_term":"TransitionModel","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prev","ontology_term":"PreviousState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"a","ontology_term":"Action","description":null}]} *)
