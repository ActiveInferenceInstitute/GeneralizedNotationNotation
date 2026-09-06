-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Hidden Markov Model Baseline
-- A standard discrete Hidden Markov Model with:
- 4 hidden states with Markovian dynamics
- 6 observation symbols
- Fixed transition and emission matrices
- No action selection (passive inference only)
- Suitable for sequence modeling and state estimation tasks
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "HiddenMarkovModelBaseline"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Hidden Markov Model Baseline"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "A standard discrete Hidden Markov Model with:\n- 4 hidden states with Markovian dynamics\n- 6 observation symbols\n- Fixed transition and emission matrices\n- No action selection (passive inference only)\n- Suitable for sequence modeling and state estimation tasks"

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 6, GnnDim.lit 4], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B", dims := [GnnDim.lit 4, GnnDim.lit 4], valueType := GnnValueType.floatT, defaultValue := none }, { name := "D", dims := [GnnDim.lit 4], valueType := GnnValueType.floatT, defaultValue := none }, { name := "F", dims := [GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "alpha", dims := [GnnDim.lit 4, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "beta", dims := [GnnDim.lit 4, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "o", dims := [GnnDim.lit 6, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s", dims := [GnnDim.lit 4, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "s_prime", dims := [GnnDim.lit 4, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "t", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "D", kind := ConnKind.directed, dst := "s", label := none }, { src := "s", kind := ConnKind.undirected, dst := "A", label := none }, { src := "s", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "A", kind := ConnKind.undirected, dst := "o", label := none }, { src := "B", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "s", kind := ConnKind.undirected, dst := "B", label := none }, { src := "s", kind := ConnKind.undirected, dst := "F", label := none }, { src := "o", kind := ConnKind.undirected, dst := "F", label := none }, { src := "s", kind := ConnKind.undirected, dst := "alpha", label := none }, { src := "o", kind := ConnKind.undirected, dst := "alpha", label := none }, { src := "alpha", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "s_prime", kind := ConnKind.undirected, dst := "beta", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }, { key := "horizon", value := some "Unbounded" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "EmissionMatrix" }, { varName := "B", term := "TransitionMatrix" }, { varName := "D", term := "InitialStateDistribution" }, { varName := "s", term := "HiddenState" }, { varName := "s_prime", term := "NextHiddenState" }, { varName := "o", term := "Observation" }, { varName := "F", term := "VariationalFreeEnergy" }, { varName := "alpha", term := "ForwardVariable" }, { varName := "beta", term := "BackwardVariable" }, { varName := "t", term := "Time" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.4, 0.4], [0.4, 0.4, 0.1, 0.1]]" }, { key := "B", value := "[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.2, 0.1], [0.1, 0.1, 0.6, 0.2], [0.1, 0.1, 0.1, 0.6]]" }, { key := "D", value := "[[0.25, 0.25, 0.25, 0.25]]" }, { key := "num_hidden_states", value := "4" }, { key := "num_observations", value := "6" }, { key := "num_timesteps", value := "50" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Hidden Markov Model Baseline emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[6,4],"value_type":"float"},{"decl":"B","dims":[4,4],"value_type":"float"},{"decl":"D","dims":[4],"value_type":"float"},{"decl":"F","dims":[1],"value_type":"float"},{"decl":"alpha","dims":[4,1],"value_type":"float"},{"decl":"beta","dims":[4,1],"value_type":"float"},{"decl":"o","dims":[6,1],"value_type":"integer"},{"decl":"s","dims":[4,1],"value_type":"float"},{"decl":"s_prime","dims":[4,1],"value_type":"float"},{"decl":"t","dims":[1],"value_type":"integer"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"EmissionMatrix"},{"var_name":"B","term":"TransitionMatrix"},{"var_name":"D","term":"InitialStateDistribution"},{"var_name":"s","term":"HiddenState"},{"var_name":"s_prime","term":"NextHiddenState"},{"var_name":"o","term":"Observation"},{"var_name":"F","term":"VariationalFreeEnergy"},{"var_name":"alpha","term":"ForwardVariable"},{"var_name":"beta","term":"BackwardVariable"},{"var_name":"t","term":"Time"}],"model_name":"Hidden Markov Model Baseline","annotation":"A standard discrete Hidden Markov Model with:\n- 4 hidden states with Markovian dynamics\n- 6 observation symbols\n- Fixed transition and emission matrices\n- No action selection (passive inference only)\n- Suitable for sequence modeling and state estimation tasks","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[6,4]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[4,4]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[4]},{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"alpha","var_type":"action","data_type":"float","dimensions":[4,1]},{"name":"beta","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[6,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"annotation":null,"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["F"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o"],"target_variables":["F"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["alpha"],"connection_type":"undirected"},{"annotation":null,"source_variables":["o"],"target_variables":["alpha"],"connection_type":"undirected"},{"annotation":null,"source_variables":["alpha"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_prime"],"target_variables":["beta"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.7,0.1,0.1,0.1],[0.1,0.7,0.1,0.1],[0.1,0.1,0.7,0.1],[0.1,0.1,0.1,0.7],[0.1,0.1,0.4,0.4],[0.4,0.4,0.1,0.1]],"param_type":"constant"},{"name":"B","value":[[0.7,0.1,0.1,0.1],[0.1,0.7,0.2,0.1],[0.1,0.1,0.6,0.2],[0.1,0.1,0.1,0.6]],"param_type":"constant"},{"name":"D","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"num_hidden_states","value":4,"param_type":"constant"},{"name":"num_observations","value":6,"param_type":"constant"},{"name":"num_timesteps","value":50,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"EmissionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"InitialStateDistribution","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"F","ontology_term":"VariationalFreeEnergy","description":null},{"variable_name":"alpha","ontology_term":"ForwardVariable","description":null},{"variable_name":"beta","ontology_term":"BackwardVariable","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
