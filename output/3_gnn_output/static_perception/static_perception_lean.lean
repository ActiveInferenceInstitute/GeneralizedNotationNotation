-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Static Perception Model
-- The simplest Active Inference model demonstrating pure perception:

- 2 hidden states mapped to 2 observations via a recognition matrix A
- Prior D encodes initial beliefs over hidden states
- Minimal 2-action transition component B so the model is a complete POMDP
  (renderable and executable by pymdp and the general simulation frameworks)
- Suitable as a minimal baseline and for testing perception-only inference
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "StaticPerceptionModel"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Static Perception Model"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "The simplest Active Inference model demonstrating pure perception:\n\n- 2 hidden states mapped to 2 observations via a recognition matrix A\n- Prior D encodes initial beliefs over hidden states\n- Minimal 2-action transition component B so the model is a complete POMDP\n  (renderable and executable by pymdp and the general simulation frameworks)\n- Suitable as a minimal baseline and for testing perception-only inference"

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B", dims := [GnnDim.lit 2, GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "C", dims := [GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "D", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "o", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "u", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "D", kind := ConnKind.directed, dst := "s", label := none }, { src := "s", kind := ConnKind.undirected, dst := "A", label := none }, { src := "A", kind := ConnKind.undirected, dst := "o", label := none }, { src := "s", kind := ConnKind.undirected, dst := "B", label := none }, { src := "B", kind := ConnKind.directed, dst := "u", label := none }, { src := "u", kind := ConnKind.directed, dst := "s", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Static" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "RecognitionMatrix" }, { varName := "B", term := "TransitionMatrix" }, { varName := "C", term := "PreferenceVector" }, { varName := "D", term := "Prior" }, { varName := "s", term := "HiddenState" }, { varName := "o", term := "Observation" }, { varName := "u", term := "Action" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[0.9, 0.1], [0.2, 0.8]]" }, { key := "B", value := "[[[0.95, 0.05], [0.05, 0.95]], [[0.05, 0.95], [0.95, 0.05]]]" }, { key := "C", value := "[[0.0, 0.0]]" }, { key := "D", value := "[[0.5, 0.5]]" }, { key := "num_hidden_states", value := "2" }, { key := "num_obs", value := "2" }, { key := "num_actions", value := "2" }, { key := "num_timesteps", value := "5" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Static Perception Model emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[2,2],"value_type":"float"},{"decl":"B","dims":[2,2,2],"value_type":"float"},{"decl":"C","dims":[2],"value_type":"float"},{"decl":"D","dims":[2,1],"value_type":"float"},{"decl":"o","dims":[2,1],"value_type":"integer"},{"decl":"s","dims":[2,1],"value_type":"float"},{"decl":"u","dims":[1],"value_type":"integer"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"RecognitionMatrix"},{"var_name":"B","term":"TransitionMatrix"},{"var_name":"C","term":"PreferenceVector"},{"var_name":"D","term":"Prior"},{"var_name":"s","term":"HiddenState"},{"var_name":"o","term":"Observation"},{"var_name":"u","term":"Action"}],"model_name":"Static Perception Model","annotation":"The simplest Active Inference model demonstrating pure perception:\n\n- 2 hidden states mapped to 2 observations via a recognition matrix A\n- Prior D encodes initial beliefs over hidden states\n- Minimal 2-action transition component B so the model is a complete POMDP\n  (renderable and executable by pymdp and the general simulation frameworks)\n- Suitable as a minimal baseline and for testing perception-only inference","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"annotation":null,"source_variables":["B"],"target_variables":["u"],"connection_type":"directed"},{"annotation":null,"source_variables":["u"],"target_variables":["s"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[[0.95,0.05],[0.05,0.95]],[[0.05,0.95],[0.95,0.05]]],"param_type":"constant"},{"name":"C","value":[[0.0,0.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":2,"param_type":"constant"},{"name":"num_obs","value":2,"param_type":"constant"},{"name":"num_actions","value":2,"param_type":"constant"},{"name":"num_timesteps","value":5,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Static","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"C","ontology_term":"PreferenceVector","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"u","ontology_term":"Action","description":null}]}
