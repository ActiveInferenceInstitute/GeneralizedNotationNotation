(* GNN Model: NEST D03 Synthetic 2-State LGSSM *)
(* Passive linear-Gaussian state-space model (no control input) generating the
summary indices of the D03 STURM wrapper blanket over T coupling iterations.
- Hidden state x = (renovation_rate_dev, price_pressure), dimensionless deviations.
- Observation y = (price_index, demand_index, stock_index, turnover_index): relative indices of fuel prices p
  (sensory, into STURM), final energy demand d (active, out of STURM), the
  archetype stock aggregate and the turnover aggregate against fixed reference
  values (see the fixture JSON, `blanket.references`).
- q_agg/r_agg are internal aggregates carried in the fixture for posterior
  checks only; they never cross the blanket.
- There is NO counterpart to D02's cap: the cap is the Q-13 HIERARCHY imposed
  constraint of the MESSAGEix side; STURM receives none.
Deliverable D03 (AII); mirror of the D04 fixture under the D03 blanket. *)

Require Import Reals.
Require Import List.

Module NESTD03Synthetic2StateLGSSM.

(* Variables *)
Parameter F : R.
Parameter H : R.
Parameter Q : R.
Parameter R : R.
Parameter prior_cov : R.
Parameter prior_mean : R.
Parameter t : Z.
Parameter x : R.
Parameter y : R.

End NESTD03Synthetic2StateLGSSM.
(* MODEL_DATA: {"model_name":"NEST D03 Synthetic 2-State LGSSM","annotation":"Passive linear-Gaussian state-space model (no control input) generating the\nsummary indices of the D03 STURM wrapper blanket over T coupling iterations.\n- Hidden state x = (renovation_rate_dev, price_pressure), dimensionless deviations.\n- Observation y = (price_index, demand_index, stock_index, turnover_index): relative indices of fuel prices p\n  (sensory, into STURM), final energy demand d (active, out of STURM), the\n  archetype stock aggregate and the turnover aggregate against fixed reference\n  values (see the fixture JSON, `blanket.references`).\n- q_agg/r_agg are internal aggregates carried in the fixture for posterior\n  checks only; they never cross the blanket.\n- There is NO counterpart to D02's cap: the cap is the Q-13 HIERARCHY imposed\n  constraint of the MESSAGEix side; STURM receives none.\nDeliverable D03 (AII); mirror of the D04 fixture under the D03 blanket.","variables":[{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[4,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[4,4]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[0.8,-0.15],[0.05,0.9]],"param_type":"constant"},{"name":"H","value":[[-0.4,0.7],[0.5,-0.3],[0.6,0.0],[0.8,0.0]],"param_type":"constant"},{"name":"Q","value":[[0.0036,-0.0003],[-0.0003,0.0049]],"param_type":"constant"},{"name":"R","value":[[0.0016,0.0,0.0,0.0],[0.0,0.0025,0.0,0.0],[0.0,0.0,0.0009,0.0],[0.0,0.0,0.0,0.0012]],"param_type":"constant"},{"name":"prior_mean","value":[[0.02,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[0.02,0.0],[0.0,0.02]],"param_type":"constant"},{"name":"num_timesteps","value":24,"param_type":"constant"},{"name":"random_seed","value":20260911,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":4,"param_type":"constant"},{"name":"num_regions","value":1,"param_type":"constant"},{"name":"num_fuel_types","value":2,"param_type":"constant"},{"name":"num_time_slices","value":12,"param_type":"constant"},{"name":"num_archetypes","value":4,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":24,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]} *)
