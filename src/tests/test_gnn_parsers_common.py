"""Tests for utility functions and classes in gnn/parsers/common.py."""
from gnn.parsers.common import (
    normalize_variable_name,
    parse_dimensions,
    infer_variable_type,
    parse_connection_operator,
    GNNInternalRepresentation,
    Variable, Connection,
    VariableType, DataType, ConnectionType,
)


# ── normalize_variable_name ────────────────────────────────────────────────

class TestNormalizeVariableName:
    def test_unicode_pi(self):
        assert normalize_variable_name('π') == 'π'

    def test_ascii_pi_lowercase(self):
        assert normalize_variable_name('pi') == 'π'

    def test_ascii_pi_uppercase(self):
        assert normalize_variable_name('PI') == 'π'

    def test_active_inference_uppercase_preserved(self):
        for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            assert normalize_variable_name(name) == name

    def test_strips_whitespace(self):
        assert normalize_variable_name('  s1  ') == 's1'

    def test_arbitrary_lowercase_unchanged(self):
        assert normalize_variable_name('my_var') == 'my_var'

    def test_empty_string(self):
        assert normalize_variable_name('') == ''


# ── parse_dimensions ───────────────────────────────────────────────────────

class TestParseDimensions:
    def test_single_dimension(self):
        assert parse_dimensions('[3]') == [3]

    def test_multiple_dimensions(self):
        assert parse_dimensions('[2,3,4]') == [2, 3, 4]

    def test_empty_brackets(self):
        assert parse_dimensions('[]') == []

    def test_type_annotation_stops_parsing(self):
        assert parse_dimensions('[3,3,type=float]') == [3, 3]

    def test_symbolic_dimension_defaults_to_one(self):
        assert parse_dimensions('[N,3]') == [1, 3]

    def test_no_brackets(self):
        # Graceful fallback: strip-no-op then parse
        assert parse_dimensions('3') == [3]

    def test_malformed_returns_default(self):
        # Non-parseable input falls back to [1]
        result = parse_dimensions('[abc,def,type=x]')
        assert result == [1, 1]


# ── infer_variable_type ────────────────────────────────────────────────────

class TestInferVariableType:
    def test_pi_unicode(self):
        assert infer_variable_type('π') == VariableType.POLICY

    def test_pi_ascii(self):
        assert infer_variable_type('pi') == VariableType.POLICY

    def test_A_is_likelihood(self):
        assert infer_variable_type('A') == VariableType.LIKELIHOOD_MATRIX

    def test_B_is_transition(self):
        assert infer_variable_type('B') == VariableType.TRANSITION_MATRIX

    def test_C_is_preference(self):
        assert infer_variable_type('C') == VariableType.PREFERENCE_VECTOR

    def test_D_is_prior(self):
        assert infer_variable_type('D') == VariableType.PRIOR_VECTOR

    def test_G_is_policy(self):
        assert infer_variable_type('G') == VariableType.POLICY

    def test_s_prefix_is_hidden_state(self):
        assert infer_variable_type('s1') == VariableType.HIDDEN_STATE

    def test_o_prefix_is_observation(self):
        assert infer_variable_type('obs_t') == VariableType.OBSERVATION

    def test_u_prefix_is_action(self):
        assert infer_variable_type('u') == VariableType.ACTION

    def test_a_prefix_is_action(self):
        assert infer_variable_type('action_t') == VariableType.ACTION

    def test_keyword_state_in_name(self):
        assert infer_variable_type('hidden_state_factor') == VariableType.HIDDEN_STATE

    def test_keyword_obs_in_name(self):
        assert infer_variable_type('my_obs_var') == VariableType.OBSERVATION

    def test_keyword_policy_in_name(self):
        assert infer_variable_type('policy_var') == VariableType.POLICY

    def test_keyword_prior_in_name(self):
        assert infer_variable_type('prior_belief') == VariableType.PRIOR_VECTOR

    def test_keyword_likelihood_in_name(self):
        assert infer_variable_type('likelihood_mat') == VariableType.LIKELIHOOD_MATRIX

    def test_keyword_preference_in_name(self):
        assert infer_variable_type('preference_vec') == VariableType.PREFERENCE_VECTOR

    def test_unknown_defaults_to_hidden_state(self):
        assert infer_variable_type('xyz_unknown') == VariableType.HIDDEN_STATE


# ── parse_connection_operator ──────────────────────────────────────────────

class TestParseConnectionOperator:
    def test_gt_is_directed(self):
        assert parse_connection_operator('>') == ConnectionType.DIRECTED

    def test_arrow_is_directed(self):
        assert parse_connection_operator('->') == ConnectionType.DIRECTED

    def test_dash_is_undirected(self):
        assert parse_connection_operator('-') == ConnectionType.UNDIRECTED

    def test_pipe_is_conditional(self):
        assert parse_connection_operator('|') == ConnectionType.CONDITIONAL

    def test_double_arrow_is_bidirectional(self):
        assert parse_connection_operator('<->') == ConnectionType.BIDIRECTIONAL

    def test_whitespace_stripped(self):
        assert parse_connection_operator('  ->  ') == ConnectionType.DIRECTED

    def test_unknown_defaults_to_directed(self):
        assert parse_connection_operator('???') == ConnectionType.DIRECTED


# ── GNNInternalRepresentation.validate_structure ──────────────────────────

class TestGNNInternalRepresentationValidation:
    def _make_var(self, name: str) -> Variable:
        return Variable(name=name, var_type=VariableType.HIDDEN_STATE,
                        data_type=DataType.CATEGORICAL)

    def _make_conn(self, src: str, tgt: str) -> Connection:
        return Connection(source_variables=[src], target_variables=[tgt],
                          connection_type=ConnectionType.DIRECTED)

    def test_valid_empty_model(self):
        model = GNNInternalRepresentation(model_name="M")
        issues = model.validate_structure()
        assert issues == []

    def test_missing_model_name(self):
        model = GNNInternalRepresentation(model_name="")
        issues = model.validate_structure()
        assert any("Model name" in i for i in issues)

    def test_duplicate_variable_names(self):
        v1 = self._make_var('s')
        v2 = self._make_var('s')
        model = GNNInternalRepresentation(model_name="M", variables=[v1, v2])
        issues = model.validate_structure()
        assert any("unique" in i.lower() for i in issues)

    def test_connection_to_unknown_variable(self):
        v = self._make_var('s')
        conn = self._make_conn('s', 'ghost')  # 'ghost' not in variables
        model = GNNInternalRepresentation(model_name="M", variables=[v], connections=[conn])
        issues = model.validate_structure()
        assert any("ghost" in i for i in issues)

    def test_valid_model_with_connection(self):
        v1 = self._make_var('s')
        v2 = self._make_var('o')
        conn = self._make_conn('s', 'o')
        model = GNNInternalRepresentation(model_name="M", variables=[v1, v2],
                                          connections=[conn])
        assert model.validate_structure() == []
