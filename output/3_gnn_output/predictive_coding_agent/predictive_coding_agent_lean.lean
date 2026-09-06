-- Canonical: FEP.GnnDocument (fep_lean v0.5)
-- Model: Predictive Coding Active Inference Agent
-- A continuous predictive-coding Active Inference agent rendered as a native
linear-Gaussian state-space model (LGSSM). The agent runs passively — it has no
control input:
- Hidden state mu = (mu, mu_dot): the generalized coordinates of the belief.
- Observation y: an identity readout of both generalized coordinates.
- F encodes the linearized dynamics f(mu): mu integrates mu_dot (dt = 0.1) and
  mu_dot leaks toward the flow.
- Q and R are the dynamics- and sensory-error covariances (the inverse
  precisions of the predictive-coding formulation).
import FepSketches.gnn_document

open FEP.GnnDocument

-- GnnSectionKind.gnnSection
def _gnnSection : GnnSection := .gnnSection "PredictiveCodingActiveInferenceAgent"

-- GnnSectionKind.gnnVersionAndFlags
def _gnnVersionAndFlags : GnnSection := .gnnVersionAndFlags GnnVersion.v1_0 []

-- GnnSectionKind.modelName
def _modelName : GnnSection := .modelName "Predictive Coding Active Inference Agent"

-- GnnSectionKind.modelAnnotation
def _modelAnnotation : GnnSection := .modelAnnotation "A continuous predictive-coding Active Inference agent rendered as a native\nlinear-Gaussian state-space model (LGSSM). The agent runs passively — it has no\ncontrol input:\n- Hidden state mu = (mu, mu_dot): the generalized coordinates of the belief.\n- Observation y: an identity readout of both generalized coordinates.\n- F encodes the linearized dynamics f(mu): mu integrates mu_dot (dt = 0.1) and\n  mu_dot leaks toward the flow.\n- Q and R are the dynamics- and sensory-error covariances (the inverse\n  precisions of the predictive-coding formulation)."

-- GnnSectionKind.stateSpaceBlock
def _stateSpaceBlock : GnnSection := .stateSpaceBlock [{ name := "F", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "H", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "Q", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "R", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "prior_cov", dims := [GnnDim.lit 2, GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "prior_mean", dims := [GnnDim.lit 2], valueType := GnnValueType.floatT, defaultValue := none }, { name := "t", dims := [GnnDim.lit 1], valueType := GnnValueType.intT, defaultValue := none }, { name := "x", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }, { name := "y", dims := [GnnDim.lit 2, GnnDim.lit 1], valueType := GnnValueType.floatT, defaultValue := none }]

-- GnnSectionKind.connections
def _connections : GnnSection := .connections [{ src := "prior_mean", kind := ConnKind.directed, dst := "x", label := none }, { src := "F", kind := ConnKind.directed, dst := "x", label := none }, { src := "x", kind := ConnKind.directed, dst := "y", label := none }, { src := "H", kind := ConnKind.directed, dst := "y", label := none }, { src := "Q", kind := ConnKind.directed, dst := "x", label := none }, { src := "R", kind := ConnKind.directed, dst := "y", label := none }]

-- GnnSectionKind.initialParameterization
def _initialParameterization : GnnSection := .initialParameterization []

-- GnnSectionKind.equations
def _equations : GnnSection := .equations ""

-- GnnSectionKind.time
def _time : GnnSection := .time [{ key := "time_type", value := some "Dynamic" }, { key := "horizon", value := some "15" }]

-- GnnSectionKind.actInfOntologyAnnotation
def _actInfOntologyAnnotation : GnnSection := .actInfOntologyAnnotation [{ varName := "F", term := "StateTransitionMatrix" }, { varName := "H", term := "ObservationMatrix" }, { varName := "Q", term := "ProcessNoiseCovariance" }, { varName := "R", term := "ObservationNoiseCovariance" }, { varName := "prior_mean", term := "PriorMean" }, { varName := "prior_cov", term := "PriorCovariance" }, { varName := "x", term := "ContinuousHiddenState" }, { varName := "y", term := "ContinuousObservation" }, { varName := "t", term := "Time" }]

-- GnnSectionKind.modelParameters
def _modelParameters : GnnSection := .modelParameters [{ key := "F", value := "[[1.0, 0.1], [0.0, 0.8]]" }, { key := "H", value := "[[1.0, 0.0], [0.0, 1.0]]" }, { key := "Q", value := "[[0.1, 0.0], [0.0, 0.1]]" }, { key := "R", value := "[[0.25, 0.0], [0.0, 0.25]]" }, { key := "prior_mean", value := "[[0.0, 0.0]]" }, { key := "prior_cov", value := "[[1.0, 0.0], [0.0, 1.0]]" }, { key := "num_timesteps", value := "15" }, { key := "dt", value := "0.1" }, { key := "random_seed", value := "42" }, { key := "num_states", value := "2" }, { key := "num_observations", value := "2" }]

-- GnnSectionKind.footer
def _footer : GnnSection := .footer "GNN model Predictive Coding Active Inference Agent emitted as the canonical FEP.GnnDocument typed surface by gnn.parsers.lean_serializer."

-- GnnSectionKind.signature
def _signature : GnnSection := .signature "serializer=gnn.parsers.lean_serializer; contract=fep_lean bridge v0.5; model_family=continuous"

def document : GnnDocument :=
  { sections := [_gnnSection, _gnnVersionAndFlags, _modelName, _modelAnnotation, _stateSpaceBlock, _connections, _initialParameterization, _equations, _time, _actInfOntologyAnnotation, _modelParameters, _footer, _signature] }

-- MODEL_DATA: {"schema_version":1,"model_family":"continuous","state_spaces":[{"decl":"F","dims":[2,2],"value_type":"float"},{"decl":"H","dims":[2,2],"value_type":"float"},{"decl":"Q","dims":[2,2],"value_type":"float"},{"decl":"R","dims":[2,2],"value_type":"float"},{"decl":"prior_cov","dims":[2,2],"value_type":"float"},{"decl":"prior_mean","dims":[2],"value_type":"float"},{"decl":"t","dims":[1],"value_type":"integer"},{"decl":"x","dims":[2,1],"value_type":"float"},{"decl":"y","dims":[2,1],"value_type":"float"}],"parameterizations":[],"ontology_bindings":[{"var_name":"F","term":"StateTransitionMatrix"},{"var_name":"H","term":"ObservationMatrix"},{"var_name":"Q","term":"ProcessNoiseCovariance"},{"var_name":"R","term":"ObservationNoiseCovariance"},{"var_name":"prior_mean","term":"PriorMean"},{"var_name":"prior_cov","term":"PriorCovariance"},{"var_name":"x","term":"ContinuousHiddenState"},{"var_name":"y","term":"ContinuousObservation"},{"var_name":"t","term":"Time"}],"model_name":"Predictive Coding Active Inference Agent","annotation":"A continuous predictive-coding Active Inference agent rendered as a native\nlinear-Gaussian state-space model (LGSSM). The agent runs passively — it has no\ncontrol input:\n- Hidden state mu = (mu, mu_dot): the generalized coordinates of the belief.\n- Observation y: an identity readout of both generalized coordinates.\n- F encodes the linearized dynamics f(mu): mu integrates mu_dot (dt = 0.1) and\n  mu_dot leaks toward the flow.\n- Q and R are the dynamics- and sensory-error covariances (the inverse\n  precisions of the predictive-coding formulation).","variables":[{"name":"F","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"H","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"Q","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"R","var_type":"hidden_state","data_type":"float","dimensions":[2,2]},{"name":"prior_cov","var_type":"prior_vector","data_type":"float","dimensions":[2,2]},{"name":"prior_mean","var_type":"prior_vector","data_type":"float","dimensions":[2]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]},{"name":"x","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"y","var_type":"hidden_state","data_type":"float","dimensions":[2,1]}],"connections":[{"annotation":null,"source_variables":["prior_mean"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["F"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["x"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["H"],"target_variables":["y"],"connection_type":"directed"},{"annotation":null,"source_variables":["Q"],"target_variables":["x"],"connection_type":"directed"},{"annotation":null,"source_variables":["R"],"target_variables":["y"],"connection_type":"directed"}],"parameters":[{"name":"F","value":[[1.0,0.1],[0.0,0.8]],"param_type":"constant"},{"name":"H","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"Q","value":[[0.1,0.0],[0.0,0.1]],"param_type":"constant"},{"name":"R","value":[[0.25,0.0],[0.0,0.25]],"param_type":"constant"},{"name":"prior_mean","value":[[0.0,0.0]],"param_type":"constant"},{"name":"prior_cov","value":[[1.0,0.0],[0.0,1.0]],"param_type":"constant"},{"name":"num_timesteps","value":15,"param_type":"constant"},{"name":"dt","value":0.1,"param_type":"constant"},{"name":"random_seed","value":42,"param_type":"constant"},{"name":"num_states","value":2,"param_type":"constant"},{"name":"num_observations","value":2,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":15,"step_size":null},"ontology_mappings":[{"variable_name":"F","ontology_term":"StateTransitionMatrix","description":null},{"variable_name":"H","ontology_term":"ObservationMatrix","description":null},{"variable_name":"Q","ontology_term":"ProcessNoiseCovariance","description":null},{"variable_name":"R","ontology_term":"ObservationNoiseCovariance","description":null},{"variable_name":"prior_mean","ontology_term":"PriorMean","description":null},{"variable_name":"prior_cov","ontology_term":"PriorCovariance","description":null},{"variable_name":"x","ontology_term":"ContinuousHiddenState","description":null},{"variable_name":"y","ontology_term":"ContinuousObservation","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
