-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Dynamic Perception Model
-- A dynamic perception model extending the static model with temporal dynamics:

- 2 hidden states evolving over discrete time via transition matrix B
- 2 observations generated from states via recognition matrix A
- Prior D constrains the initial hidden state
- No action selection — the agent passively observes a changing world
- Demonstrates belief updating (state inference) across time steps
- Suitable for tracking hidden sources from noisy observations
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "DynamicPerceptionModel"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Dynamic Perception Model"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "A dynamic perception model extending the static model with temporal dynamics:\n\n- 2 hidden states evolving over discrete time via transition matrix B\n- 2 observations generated from states via recognition matrix A\n- Prior D constrains the initial hidden state\n- No action selection — the agent passively observes a changing world\n- Demonstrates belief updating (state inference) across time steps\n- Suitable for tracking hidden sources from noisy observations"

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "D", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "F", dims := [GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "o_t", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s_prime", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "s_t", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "t", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "D", kind := ConnKind.directed, dst := "s_t", label := none }, { src := "s_t", kind := ConnKind.undirected, dst := "A", label := none }, { src := "A", kind := ConnKind.undirected, dst := "o_t", label := none }, { src := "s_t", kind := ConnKind.undirected, dst := "B", label := none }, { src := "B", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "s_t", kind := ConnKind.undirected, dst := "F", label := none }, { src := "o_t", kind := ConnKind.undirected, dst := "F", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }, { key := "horizon", value := some "10" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "RecognitionMatrix" }, { varName := "B", term := "TransitionMatrix" }, { varName := "D", term := "Prior" }, { varName := "s_t", term := "HiddenState" }, { varName := "s_prime", term := "NextHiddenState" }, { varName := "o_t", term := "Observation" }, { varName := "F", term := "VariationalFreeEnergy" }, { varName := "t", term := "Time" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[0.9, 0.1], [0.2, 0.8]]" }, { key := "B", value := "[[0.7, 0.3], [0.3, 0.7]]" }, { key := "D", value := "[[0.5, 0.5]]" }, { key := "num_hidden_states", value := "2" }, { key := "num_obs", value := "2" }, { key := "num_timesteps", value := "10" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Dynamic Perception Model emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[2,2],"value_type":"float"},{"decl":"B","dims":[2,2],"value_type":"float"},{"decl":"D","dims":[2,1],"value_type":"float"},{"decl":"F","dims":[1],"value_type":"float"},{"decl":"o_t","dims":[2,1],"value_type":"integer"},{"decl":"s_prime","dims":[2,1],"value_type":"float"},{"decl":"s_t","dims":[2,1],"value_type":"float"},{"decl":"t","dims":[1],"value_type":"integer"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"RecognitionMatrix"},{"var_name":"B","term":"TransitionMatrix"},{"var_name":"D","term":"Prior"},{"var_name":"s_t","term":"HiddenState"},{"var_name":"s_prime","term":"NextHiddenState"},{"var_name":"o_t","term":"Observation"},{"var_name":"F","term":"VariationalFreeEnergy"},{"var_name":"t","term":"Time"}],"model_name":"Dynamic Perception Model","annotation":"A dynamic perception model extending the static model with temporal dynamics:\n\n- 2 hidden states evolving over discrete time via transition matrix B\n- 2 observations generated from states via recognition matrix A\n- Prior D constrains the initial hidden state\n- No action selection — the agent passively observes a changing world\n- Demonstrates belief updating (state inference) across time steps\n- Suitable for tracking hidden sources from noisy observations","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[2,2]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[2,2]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[2,1]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"o_t","var_type":"observation","data_type":"integer","dimensions":[2,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"s_t","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s_t"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_t"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A"],"target_variables":["o_t"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s_t"],"target_variables":["B"],"connection_type":"undirected"},{"annotation":null,"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_t"],"target_variables":["F"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o_t"],"target_variables":["F"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.9,0.1],[0.2,0.8]],"param_type":"constant"},{"name":"B","value":[[0.7,0.3],[0.3,0.7]],"param_type":"constant"},{"name":"D","value":[[0.5,0.5]],"param_type":"constant"},{"name":"num_hidden_states","value":2,"param_type":"constant"},{"name":"num_obs","value":2,"param_type":"constant"},{"name":"num_timesteps","value":10,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":10,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"RecognitionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s_t","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o_t","ontology_term":"Observation","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
