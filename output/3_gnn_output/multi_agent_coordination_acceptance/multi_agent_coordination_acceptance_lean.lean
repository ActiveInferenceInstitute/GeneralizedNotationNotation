-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Multi-Agent Coordination Acceptance Fixture
-- Compact fixture for RxInfer and DisCoPy roadmap acceptance tests.
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "MultiAgentCoordinationAcceptanceFixture"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Multi-Agent Coordination Acceptance Fixture"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "Compact fixture for RxInfer and DisCoPy roadmap acceptance tests."

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "o", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "u", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "s", kind := ConnKind.directed, dst := "o", label := none }, { src := "s", kind := ConnKind.directed, dst := "s", label := none }, { src := "u", kind := ConnKind.directed, dst := "s", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "s", term := "HiddenState" }, { varName := "o", term := "Observation" }, { varName := "u", term := "Action" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "nr_agents", value := "3" }, { key := "agent_ids", value := "[1, 2, 3]" }, { key := "agent_initial_positions", value := "[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]" }, { key := "agent_target_positions", value := "[[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]]" }, { key := "agent_radii", value := "[1.0, 1.0, 1.0]" }, { key := "agent_edges", value := "[[1, 2], [2, 3]]" }, { key := "agent_clusters", value := "[{'name': 'left', 'agent_ids': [1, 2]}, {'name': 'right', 'agent_ids': [3]}]" }, { key := "message_passing", value := "clustered_mean_field" }, { key := "A", value := "[[0.9, 0.1], [0.1, 0.9]]" }, { key := "B", value := "[[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]" }, { key := "C", value := "[[1.0, 0.0]]" }, { key := "D", value := "[[0.5, 0.5]]" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Multi-Agent Coordination Acceptance Fixture emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"o","dims":[2,1],"value_type":"categorical"},{"decl":"s","dims":[2,1],"value_type":"categorical"},{"decl":"u","dims":[2,1],"value_type":"categorical"}],"parameterizations":[],"ontology_bindings":[{"var_name":"s","term":"HiddenState"},{"var_name":"o","term":"Observation"},{"var_name":"u","term":"Action"}],"model_name":"Multi-Agent Coordination Acceptance Fixture","annotation":"Compact fixture for RxInfer and DisCoPy roadmap acceptance tests.","variables":[{"name":"o","var_type":"observation","data_type":"categorical","dimensions":[2,1]},{"name":"s","var_type":"hidden_state","data_type":"categorical","dimensions":[2,1]},{"name":"u","var_type":"action","data_type":"categorical","dimensions":[2,1]}],"connections":[{"annotation":null,"source_variables":["s"],"target_variables":["o"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["u"],"target_variables":["s"],"connection_type":"directed"}],"parameters":[{"name":"nr_agents","value":3,"param_type":"constant"},{"name":"agent_ids","value":[1,2,3],"param_type":"constant"},{"name":"agent_initial_positions","value":[[0.0,0.0],[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"agent_target_positions","value":[[2.0,2.0],[3.0,2.0],[2.0,3.0]],"param_type":"constant"},{"name":"agent_radii","value":[1.0,1.0,1.0],"param_type":"constant"},{"name":"agent_edges","value":[[1,2],[2,3]],"param_type":"constant"},{"name":"agent_clusters","value":[{"name":"left","agent_ids":[1,2]},{"name":"right","agent_ids":[3]}],"param_type":"constant"},{"name":"message_passing","value":"clustered_mean_field","param_type":"constant"},{"name":"A","value":[[0.9,0.1],[0.1,0.9]],"param_type":"constant"},{"name":"B","value":[[[0.9,0.1],[0.1,0.9]],[[0.1,0.9],[0.9,0.1]]],"param_type":"constant"},{"name":"C","value":[[1.0,0.0]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":null,"step_size":null},"ontology_mappings":[{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"u","ontology_term":"Action","description":null}]}
