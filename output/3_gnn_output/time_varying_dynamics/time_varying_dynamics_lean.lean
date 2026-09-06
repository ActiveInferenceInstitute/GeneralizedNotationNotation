-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Time-Varying Transition Dynamics Agent
-- A POMDP agent operating in a non-stationary environment. The key feature
is that the transition matrix `B` is indexed by time (`B_t`), capturing
dynamics that evolve across the planning horizon — e.g., shifting wind
patterns for a sailing agent, or changing opponent strategy in a
sequential game.

- 3 hidden states, 3 observations, 2 actions
- B_t: 3D transition tensor per timestep (shape: next_state × current_state × action)
- Agent must adapt belief updates each step to the current B_t
- Exercises time-varying matrix handling in renderers

This sample pushes the language extensions around time-indexed tensors
and tests downstream code generation when matrix literals are
timestep-dependent.
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "TimeVaryingTransitionDynamicsAgent"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Time-Varying Transition Dynamics Agent"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "A POMDP agent operating in a non-stationary environment. The key feature\nis that the transition matrix `B` is indexed by time (`B_t`), capturing\ndynamics that evolve across the planning horizon — e.g., shifting wind\npatterns for a sailing agent, or changing opponent strategy in a\nsequential game.\n\n- 3 hidden states, 3 observations, 2 actions\n- B_t: 3D transition tensor per timestep (shape: next_state × current_state × action)\n- Agent must adapt belief updates each step to the current B_t\n- Exercises time-varying matrix handling in renderers\n\nThis sample pushes the language extensions around time-indexed tensors\nand tests downstream code generation when matrix literals are\ntimestep-dependent."

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "A", dims := [GnnDim.lit 3, GnnDim.lit 3], valueType := GnnValueType.floatT, defaultValue := none }, { name := "B_t", dims := [GnnDim.lit 3, GnnDim.lit 3, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "C", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "D", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "o_t", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "s_t", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "s_t+1", dims := [GnnDim.lit 3, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "u_t", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "D", kind := ConnKind.directed, dst := "s_t", label := none }, { src := "s_t", kind := ConnKind.directed, dst := "B_t", label := none }, { src := "u_t", kind := ConnKind.directed, dst := "B_t", label := none }, { src := "B_t", kind := ConnKind.directed, dst := "s_t+1", label := none }, { src := "s_t", kind := ConnKind.undirected, dst := "A", label := none }, { src := "A", kind := ConnKind.undirected, dst := "o_t", label := none }, { src := "C", kind := ConnKind.undirected, dst := "o_t", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }, { key := "discretization", value := some "DiscreteTime" }, { key := "horizon", value := some "10" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "A", term := "LikelihoodMatrix" }, { varName := "B_t", term := "TimeVaryingTransitionMatrix" }, { varName := "C", term := "PreferenceVector" }, { varName := "D", term := "Prior" }, { varName := "s_t", term := "HiddenState" }, { varName := "o_t", term := "Observation" }, { varName := "u_t", term := "Action" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "A", value := "[[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]]" }, { key := "B_t", value := "[[[0.6, 0.1], [0.3, 0.1], [0.1, 0.8]], [[0.3, 0.1], [0.6, 0.6], [0.1, 0.3]], [[0.1, 0.8], [0.1, 0.3], [0.8, 0.1]]]" }, { key := "C", value := "[[0.0, 0.0, 1.0]]" }, { key := "D", value := "[[0.33, 0.33, 0.34]]" }, { key := "num_hidden_states", value := "3" }, { key := "num_obs", value := "3" }, { key := "num_actions", value := "2" }, { key := "num_timesteps", value := "10" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Time-Varying Transition Dynamics Agent emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=finite"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"finite","state_spaces":[{"decl":"A","dims":[3,3],"value_type":"float"},{"decl":"B_t","dims":[3,3,2],"value_type":"float"},{"decl":"C","dims":[3,1],"value_type":"float"},{"decl":"D","dims":[3,1],"value_type":"float"},{"decl":"o_t","dims":[3,1],"value_type":"integer"},{"decl":"s_t","dims":[3,1],"value_type":"float"},{"decl":"s_t+1","dims":[3,1],"value_type":"float"},{"decl":"u_t","dims":[2,1],"value_type":"integer"}],"parameterizations":[],"ontology_bindings":[{"var_name":"A","term":"LikelihoodMatrix"},{"var_name":"B_t","term":"TimeVaryingTransitionMatrix"},{"var_name":"C","term":"PreferenceVector"},{"var_name":"D","term":"Prior"},{"var_name":"s_t","term":"HiddenState"},{"var_name":"o_t","term":"Observation"},{"var_name":"u_t","term":"Action"}],"model_name":"Time-Varying Transition Dynamics Agent","annotation":"A POMDP agent operating in a non-stationary environment. The key feature\nis that the transition matrix `B` is indexed by time (`B_t`), capturing\ndynamics that evolve across the planning horizon — e.g., shifting wind\npatterns for a sailing agent, or changing opponent strategy in a\nsequential game.\n\n- 3 hidden states, 3 observations, 2 actions\n- B_t: 3D transition tensor per timestep (shape: next_state × current_state × action)\n- Agent must adapt belief updates each step to the current B_t\n- Exercises time-varying matrix handling in renderers\n\nThis sample pushes the language extensions around time-indexed tensors\nand tests downstream code generation when matrix literals are\ntimestep-dependent.","variables":[{"name":"A","var_type":"likelihood_matrix","data_type":"float","dimensions":[3,3]},{"name":"B_t","var_type":"hidden_state","data_type":"float","dimensions":[3,3,2]},{"name":"C","var_type":"preference_vector","data_type":"float","dimensions":[3,1]},{"name":"D","var_type":"prior_vector","data_type":"float","dimensions":[3,1]},{"name":"o_t","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"s_t","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"s_t+1","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"u_t","var_type":"action","data_type":"integer","dimensions":[2,1]}],"connections":[{"annotation":null,"source_variables":["D"],"target_variables":["s_t"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_t","u_t"],"target_variables":["B_t"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_t"],"target_variables":["s_t+1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_t"],"target_variables":["A"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A"],"target_variables":["o_t"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C"],"target_variables":["o_t"],"connection_type":"undirected"}],"parameters":[{"name":"A","value":[[0.85,0.1,0.05],[0.1,0.8,0.1],[0.05,0.1,0.85]],"param_type":"constant"},{"name":"B_t","value":[[[0.6,0.1],[0.3,0.1],[0.1,0.8]],[[0.3,0.1],[0.6,0.6],[0.1,0.3]],[[0.1,0.8],[0.1,0.3],[0.8,0.1]]],"param_type":"constant"},{"name":"C","value":[[0.0,0.0,1.0]],"param_type":"constant"},{"name":"D","value":[[0.33,0.33,0.34]],"param_type":"constant"},{"name":"num_hidden_states","value":3,"param_type":"constant"},{"name":"num_obs","value":3,"param_type":"constant"},{"name":"num_actions","value":2,"param_type":"constant"},{"name":"num_timesteps","value":10,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":"DiscreteTime","horizon":10,"step_size":null},"ontology_mappings":[{"variable_name":"A","ontology_term":"LikelihoodMatrix","description":null},{"variable_name":"B_t","ontology_term":"TimeVaryingTransitionMatrix","description":null},{"variable_name":"C","ontology_term":"PreferenceVector","description":null},{"variable_name":"D","ontology_term":"Prior","description":null},{"variable_name":"s_t","ontology_term":"HiddenState","description":null},{"variable_name":"o_t","ontology_term":"Observation","description":null},{"variable_name":"u_t","ontology_term":"Action","description":null}]}
