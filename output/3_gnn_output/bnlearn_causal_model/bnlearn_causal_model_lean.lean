-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Bnlearn Causal Model
-- A Bayesian Network model mapping Active Inference structure:
- S: Hidden State
- A: Action
- S_prev: Previous State
- O: Observation
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "BnlearnCausalModel"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Bnlearn Causal Model"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "A Bayesian Network model mapping Active Inference structure:\n- S: Hidden State\n- A: Action\n- S_prev: Previous State\n- O: Observation"

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B", dims := [GnnDim.lit 2, GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "a", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "o", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "s_prev", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "s_prev", kind := ConnKind.directed, dst := "s", label := none }, { src := "a", kind := ConnKind.directed, dst := "s", label := none }, { src := "s", kind := ConnKind.directed, dst := "o", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "ObservationModel" }, { varName := "B", term := "TransitionModel" }, { varName := "s", term := "HiddenState" }, { varName := "s_prev", term := "PreviousState" }, { varName := "o", term := "Observation" }, { varName := "a", term := "Action" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[0.9, 0.1], [0.1, 0.9]]" }, { key := "B", value := "[[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]]" }, { key := "C", value := "[[0.0, 1.0]]" }, { key := "D", value := "[[0.5, 0.5]]" }, { key := "num_timesteps", value := "30" }, { key := "num_hidden_states", value := "2" }, { key := "num_obs", value := "2" }, { key := "num_actions", value := "2" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Bnlearn Causal Model emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[2,2],"value_type":"float"},{"decl":"B","dims":[2,2,2],"value_type":"float"},{"decl":"a","dims":[2,1],"value_type":"integer"},{"decl":"o","dims":[2,1],"value_type":"integer"},{"decl":"s","dims":[2,1],"value_type":"float"},{"decl":"s_prev","dims":[2,1],"value_type":"float"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"ObservationModel"},{"var_name":"B","term":"TransitionModel"},{"var_name":"s","term":"HiddenState"},{"var_name":"s_prev","term":"PreviousState"},{"var_name":"o","term":"Observation"},{"var_name":"a","term":"Action"}],"model_name":"Bnlearn Causal Model","annotation":"A Bayesian Network model mapping Active Inference structure:\n- S: Hidden State\n- A: Action\n- S_prev: Previous State\n- O: Observation","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2,2]},{"name":"a","var_type":"action","data_type":"integer","dimensions":[2,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_prev","var_type":"hidden_state","data_type":"float","dimensions":[2,1]}],"connections":[{"annotation":null,"source_variables":["s_prev"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["a"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["o"],"connection_type":"directed"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B","value":[[[0.7,0.3],[0.3,0.7]],[[0.3,0.7],[0.7,0.3]]],"param_type":"constant"},{"name":"C","value":[[0.0,1.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_timesteps","value":30,"param_type":"constant"},{"name":"num_hidden_states","value":2,"param_type":"constant"},{"name":"num_obs","value":2,"param_type":"constant"},{"name":"num_actions","value":2,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"ObservationModel","description":null},{"variable_name":"B","ontology_term":"TransitionModel","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prev","ontology_term":"PreviousState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"a","ontology_term":"Action","description":null}]}
