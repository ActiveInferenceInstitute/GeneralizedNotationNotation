-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Multi Armed Bandit Agent
-- This model describes a 3-armed bandit as a degenerate POMDP:

- 3 hidden states representing the "reward context" (which arm is currently best).
- 3 observations representing reward signals (no-reward, small-reward, big-reward).
- 3 actions: pull arm 0, pull arm 1, or pull arm 2.
- Context switches slowly (sticky transitions), testing exploration vs exploitation.
- The agent prefers big-reward observations (observation 2).
- Tests the bandit structure: meaningful actions despite nearly-static state dynamics.
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "MultiArmedBanditAgent"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Multi Armed Bandit Agent"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "This model describes a 3-armed bandit as a degenerate POMDP:\n\n- 3 hidden states representing the \"reward context\" (which arm is currently best).\n- 3 observations representing reward signals (no-reward, small-reward, big-reward).\n- 3 actions: pull arm 0, pull arm 1, or pull arm 2.\n- Context switches slowly (sticky transitions), testing exploration vs exploitation.\n- The agent prefers big-reward observations (observation 2).\n- Tests the bandit structure: meaningful actions despite nearly-static state dynamics."

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 3, GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B", dims := [GnnDim.lit 3, GnnDim.lit 3, GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "C", dims := [GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "D", dims := [GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "G", dims := [GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "o", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "s_prime", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "t", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "u", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "π", dims := [GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "D", kind := ConnKind.directed, dst := "s", label := none }, { src := "s", kind := ConnKind.undirected, dst := "A", label := none }, { src := "A", kind := ConnKind.undirected, dst := "o", label := none }, { src := "s", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "s", kind := ConnKind.undirected, dst := "B", label := none }, { src := "C", kind := ConnKind.directed, dst := "G", label := none }, { src := "G", kind := ConnKind.directed, dst := "π", label := none }, { src := "π", kind := ConnKind.directed, dst := "u", label := none }, { src := "B", kind := ConnKind.directed, dst := "u", label := none }, { src := "u", kind := ConnKind.directed, dst := "s_prime", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }, { key := "horizon", value := some "Unbounded" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "LikelihoodMatrix" }, { varName := "B", term := "TransitionMatrix" }, { varName := "C", term := "LogPreferenceVector" }, { varName := "D", term := "PriorOverHiddenStates" }, { varName := "G", term := "ExpectedFreeEnergy" }, { varName := "s", term := "HiddenState" }, { varName := "s_prime", term := "NextHiddenState" }, { varName := "o", term := "Observation" }, { varName := "π", term := "PolicyVector" }, { varName := "u", term := "Action" }, { varName := "t", term := "Time" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[0.1, 0.5, 0.5], [0.3, 0.4, 0.3], [0.6, 0.1, 0.2]]" }, { key := "B", value := "[[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]]" }, { key := "C", value := "[[0.0, 1.0, 3.0]]" }, { key := "D", value := "[[0.33333, 0.33333, 0.33333]]" }, { key := "num_hidden_states", value := "3" }, { key := "num_obs", value := "3" }, { key := "num_actions", value := "3" }, { key := "num_timesteps", value := "30" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Multi Armed Bandit Agent emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[3,3],"value_type":"float"},{"decl":"B","dims":[3,3,3],"value_type":"float"},{"decl":"C","dims":[3],"value_type":"float"},{"decl":"D","dims":[3],"value_type":"float"},{"decl":"G","dims":[1],"value_type":"float"},{"decl":"o","dims":[3,1],"value_type":"integer"},{"decl":"s","dims":[3,1],"value_type":"float"},{"decl":"s_prime","dims":[3,1],"value_type":"float"},{"decl":"t","dims":[1],"value_type":"integer"},{"decl":"u","dims":[1],"value_type":"integer"},{"decl":"π","dims":[3],"value_type":"float"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"LikelihoodMatrix"},{"var_name":"B","term":"TransitionMatrix"},{"var_name":"C","term":"LogPreferenceVector"},{"var_name":"D","term":"PriorOverHiddenStates"},{"var_name":"G","term":"ExpectedFreeEnergy"},{"var_name":"s","term":"HiddenState"},{"var_name":"s_prime","term":"NextHiddenState"},{"var_name":"o","term":"Observation"},{"var_name":"π","term":"PolicyVector"},{"var_name":"u","term":"Action"},{"var_name":"t","term":"Time"}],"model_name":"Multi Armed Bandit Agent","annotation":"This model describes a 3-armed bandit as a degenerate POMDP:\n\n- 3 hidden states representing the \"reward context\" (which arm is currently best).\n- 3 observations representing reward signals (no-reward, small-reward, big-reward).\n- 3 actions: pull arm 0, pull arm 1, or pull arm 2.\n- Context switches slowly (sticky transitions), testing exploration vs exploitation.\n- The agent prefers big-reward observations (observation 2).\n- Tests the bandit structure: meaningful actions despite nearly-static state dynamics.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3,3]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"G","var_type":"policy","data_type":"float","dimensions":[1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"π","var_type":"policy","data_type":"float","dimensions":[3]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C"],"target_variables":["G"],"connection_type":"directed"},{"annotation":null,"source_variables":["G"],"target_variables":["π"],"connection_type":"directed"},{"annotation":null,"source_variables":["π"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["u"],"target_variables":["s_prime"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.1,0.5,0.5],[0.3,0.4,0.3],[0.6,0.1,0.2]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]],[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]],"param_type":"constant"},{"name":"C","value":[[0.0,1.0,3.0]],"param_type":"constant"},{"name":"D","value":[[0.33333,0.33333,0.33333]],"param_type":"constant"},{"name":"num_hidden_states","value":3,"param_type":"constant"},{"name":"num_obs","value":3,"param_type":"constant"},{"name":"num_actions","value":3,"param_type":"constant"},{"name":"num_timesteps","value":30,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"LogPreferenceVector","description":null},{"variable_name":"D","ontology_term":"PriorOverHiddenStates","description":null},{"variable_name":"G","ontology_term":"ExpectedFreeEnergy","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"π","ontology_term":"PolicyVector","description":null},{"variable_name":"u","ontology_term":"Action","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
