(* GNN Model: T-Maze Epistemic Foraging Agent *)
(* The classic T-maze task from Active Inference literature (Friston et al.):

- Agent navigates a T-shaped maze with 4 locations: center, left arm, right arm, cue location
- Two observation modalities: location (where am I?) and reward/cue (what do I see?)
- Reward is hidden behind one of the two arms (left or right), determined by context
- Cue location provides partial information about which arm holds the reward
- Agent must decide: go directly to an arm (exploit) or visit cue location first (explore)
- Demonstrates epistemic foraging: Active Inference naturally balances exploration vs exploitation
- The Expected Free Energy decomposes into epistemic (information gain) + instrumental (reward) value *)

Require Import Reals.
Require Import List.

Module TMazeEpistemicForagingAgent.

(* Variables *)
Parameter A_loc : R.
Parameter A_rew : R.
Parameter B_ctx : R.
Parameter B_loc : R.
Parameter C_loc : R.
Parameter C_rew : R.
Parameter D_ctx : R.
Parameter D_loc : R.
Parameter F : R.
Parameter G : R.
Parameter G_epi : R.
Parameter G_ins : R.
Parameter o_loc : Z.
Parameter o_rew : Z.
Parameter s_ctx : R.
Parameter s_loc : R.
Parameter t : Z.
Parameter u : Z.
Parameter π : R.

End TMazeEpistemicForagingAgent.
(* MODEL_DATA: {"model_name":"T-Maze Epistemic Foraging Agent","annotation":"The classic T-maze task from Active Inference literature (Friston et al.):\n\n- Agent navigates a T-shaped maze with 4 locations: center, left arm, right arm, cue location\n- Two observation modalities: location (where am I?) and reward/cue (what do I see?)\n- Reward is hidden behind one of the two arms (left or right), determined by context\n- Cue location provides partial information about which arm holds the reward\n- Agent must decide: go directly to an arm (exploit) or visit cue location first (explore)\n- Demonstrates epistemic foraging: Active Inference naturally balances exploration vs exploitation\n- The Expected Free Energy decomposes into epistemic (information gain) + instrumental (reward) value","variables":[{"name":"s_loc","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_ctx","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o_loc","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"o_rew","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"A_loc","var_type":"action","data_type":"float","dimensions":[4,4]},{"name":"A_rew","var_type":"action","data_type":"float","dimensions":[3,4,2]},{"name":"B_loc","var_type":"hidden_state","data_type":"float","dimensions":[4,4,4]},{"name":"B_ctx","var_type":"hidden_state","data_type":"float","dimensions":[2,2,1]},{"name":"C_loc","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"C_rew","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D_loc","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_ctx","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"\u03c0","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"G_epi","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"G_ins","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D_loc"],"target_variables":["s_loc"],"connection_type":"directed"},{"source_variables":["D_ctx"],"target_variables":["s_ctx"],"connection_type":"directed"},{"source_variables":["s_loc"],"target_variables":["A_loc"],"connection_type":"undirected"},{"source_variables":["A_loc"],"target_variables":["o_loc"],"connection_type":"undirected"},{"source_variables":["s_loc"],"target_variables":["A_rew"],"connection_type":"undirected"},{"source_variables":["s_ctx"],"target_variables":["A_rew"],"connection_type":"undirected"},{"source_variables":["A_rew"],"target_variables":["o_rew"],"connection_type":"undirected"},{"source_variables":["s_loc"],"target_variables":["B_loc"],"connection_type":"undirected"},{"source_variables":["s_ctx"],"target_variables":["B_ctx"],"connection_type":"undirected"},{"source_variables":["C_rew"],"target_variables":["G_ins"],"connection_type":"directed"},{"source_variables":["G_epi"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G_ins"],"target_variables":["G"],"connection_type":"directed"},{"source_variables":["G"],"target_variables":["\u03c0"],"connection_type":"directed"},{"source_variables":["\u03c0"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["B_loc"],"target_variables":["u"],"connection_type":"directed"},{"source_variables":["s_loc"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["s_ctx"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_loc"],"target_variables":["F"],"connection_type":"undirected"},{"source_variables":["o_rew"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A_loc","value":[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],"param_type":"constant"},{"name":"B_loc","value":[[[0.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[1.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]],[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[1.0,0.0,0.0,1.0]],[[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,1.0,1.0,0.0]]],"param_type":"constant"},{"name":"B_ctx","value":[[[1.0,0.0],[0.0,1.0]]],"param_type":"constant"},{"name":"C_loc","value":[[0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"C_rew","value":[[-1.0,3.0,0.0]],"param_type":"constant"},{"name":"D_loc","value":[[1.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D_ctx","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":3,"step_size":null},"ontology_mappings":[{"variable_name":"A_loc","ontology_term":"LocationLikelihoodMatrix","description":null},{"variable_name":"A_rew","ontology_term":"RewardLikelihoodMatrix","description":null},{"variable_name":"B_loc","ontology_term":"LocationTransitionMatrix","description":null},{"variable_name":"B_ctx","ontology_term":"ContextTransitionMatrix","description":null},{"variable_name":"C_loc","ontology_term":"LocationPreferenceVector","description":null},{"variable_name":"C_rew","ontology_term":"RewardPreferenceVector","description":null},{"variable_name":"D_loc","ontology_term":"LocationPrior","description":null},{"variable_name":"D_ctx","ontology_term":"ContextPrior","description":null},{"variable_name":"s_loc","ontology_term":"LocationHiddenState","description":null},{"variable_name":"s_ctx","ontology_term":"ContextHiddenState","description":null},{"variable_name":"o_loc","ontology_term":"LocationObservation","description":null},{"variable_name":"o_rew","ontology_term":"RewardObservation","description":null},{"variable_name":"\u03c0","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"G_epi","ontology_term":"EpistemicValue","description":null},{"variable_name":"G_ins","ontology_term":"InstrumentalValue","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]} *)
