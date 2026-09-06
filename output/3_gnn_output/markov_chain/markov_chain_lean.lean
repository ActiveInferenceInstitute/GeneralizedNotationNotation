-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Simple Markov Chain
-- This model describes a minimal discrete-time Markov Chain:

- 3 states representing weather (sunny, cloudy, rainy).
- No actions — the system evolves passively.
- Observations = states directly (identity mapping for monitoring).
- Stationary transition matrix with realistic weather dynamics.
- Tests the simplest model structure: passive state evolution with no control.
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "SimpleMarkovChain"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Simple Markov Chain"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "This model describes a minimal discrete-time Markov Chain:\n\n- 3 states representing weather (sunny, cloudy, rainy).\n- No actions — the system evolves passively.\n- Observations = states directly (identity mapping for monitoring).\n- Stationary transition matrix with realistic weather dynamics.\n- Tests the simplest model structure: passive state evolution with no control."

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 3, GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B", dims := [GnnDim.lit 3, GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "D", dims := [GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "o", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "s_prime", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "t", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "D", kind := ConnKind.directed, dst := "s", label := none }, { src := "s", kind := ConnKind.undirected, dst := "A", label := none }, { src := "A", kind := ConnKind.undirected, dst := "o", label := none }, { src := "s", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "B", kind := ConnKind.directed, dst := "s_prime", label := none }, { src := "s", kind := ConnKind.undirected, dst := "B", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }, { key := "horizon", value := some "Unbounded" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "EmissionMatrix" }, { varName := "B", term := "TransitionMatrix" }, { varName := "D", term := "InitialStateDistribution" }, { varName := "s", term := "HiddenState" }, { varName := "s_prime", term := "NextHiddenState" }, { varName := "o", term := "Observation" }, { varName := "t", term := "Time" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]" }, { key := "B", value := "[[0.7, 0.3, 0.1], [0.2, 0.4, 0.3], [0.1, 0.3, 0.6]]" }, { key := "D", value := "[[0.5, 0.3, 0.2]]" }, { key := "num_hidden_states", value := "3" }, { key := "num_obs", value := "3" }, { key := "num_actions", value := "1" }, { key := "num_timesteps", value := "40" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Simple Markov Chain emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[3,3],"value_type":"float"},{"decl":"B","dims":[3,3],"value_type":"float"},{"decl":"D","dims":[3],"value_type":"float"},{"decl":"o","dims":[3,1],"value_type":"integer"},{"decl":"s","dims":[3,1],"value_type":"float"},{"decl":"s_prime","dims":[3,1],"value_type":"float"},{"decl":"t","dims":[1],"value_type":"integer"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"EmissionMatrix"},{"var_name":"B","term":"TransitionMatrix"},{"var_name":"D","term":"InitialStateDistribution"},{"var_name":"s","term":"HiddenState"},{"var_name":"s_prime","term":"NextHiddenState"},{"var_name":"o","term":"Observation"},{"var_name":"t","term":"Time"}],"model_name":"Simple Markov Chain","annotation":"This model describes a minimal discrete-time Markov Chain:\n\n- 3 states representing weather (sunny, cloudy, rainy).\n- No actions — the system evolves passively.\n- Observations = states directly (identity mapping for monitoring).\n- Stationary transition matrix with realistic weather dynamics.\n- Tests the simplest model structure: passive state evolution with no control.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B","var_type":"transition_matrix","data_type":"float","dimensions":[3,3]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"o","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"s","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_prime","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A"],"target_variables":["o"],"connection_type":"undirected"},{"annotation":null,"source_variables":["s"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["B"],"target_variables":["s_prime"],"connection_type":"directed"},{"annotation":null,"source_variables":["s"],"target_variables":["B"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"param_type":"constant"},{"name":"B","value":[[0.7,0.3,0.1],[0.2,0.4,0.3],[0.1,0.3,0.6]],"param_type":"constant"},{"name":"D","value":[[0.5,0.3,0.2]],"param_type":"constant"},{"name":"num_hidden_states","value":3,"param_type":"constant"},{"name":"num_obs","value":3,"param_type":"constant"},{"name":"num_actions","value":1,"param_type":"constant"},{"name":"num_timesteps","value":40,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":"Unbounded","step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"EmissionMatrix","description":null},{"variable_name":"B","ontology_term":"TransitionMatrix","description":null},{"variable_name":"D","ontology_term":"InitialStateDistribution","description":null},{"variable_name":"s","ontology_term":"HiddenState","description":null},{"variable_name":"s_prime","ontology_term":"NextHiddenState","description":null},{"variable_name":"o","ontology_term":"Observation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
