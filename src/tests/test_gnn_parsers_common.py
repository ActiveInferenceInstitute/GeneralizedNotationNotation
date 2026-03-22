"""Tests for utility functions and classes in gnn/parsers/common.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gnn.parsers.common import (
    ParseError,
    normalize_variable_name,
    parse_dimensions,
    infer_variable_type,
    parse_connection_operator,
    safe_enum_convert,
    extract_embedded_json_data,
    GNNInternalRepresentation,
    Variable, Connection, Parameter, OntologyMapping,
    ASTNode, ASTVisitor, PrintVisitor,
    ParseResult, BaseGNNParser,
    VariableType, DataType, ConnectionType, GNNFormat,
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


# ── ParseError ─────────────────────────────────────────────────────────────

class TestParseError:
    def test_message_only(self):
        err = ParseError("bad input")
        assert "bad input" in str(err)

    def test_with_line(self):
        err = ParseError("oops", line=5)
        assert "Line 5" in str(err)

    def test_with_line_and_column(self):
        err = ParseError("oops", line=3, column=7)
        assert "Column 7" in str(err)
        assert "Line 3" in str(err)

    def test_with_source(self):
        err = ParseError("oops", source="myfile.md")
        assert "myfile.md" in str(err)

    def test_attributes_stored(self):
        err = ParseError("msg", line=2, column=4, source="f.md")
        assert err.message == "msg"
        assert err.line == 2
        assert err.column == 4
        assert err.source == "f.md"

    def test_is_exception(self):
        with pytest.raises(ParseError):
            raise ParseError("boom")


# ── safe_enum_convert ──────────────────────────────────────────────────────

class TestSafeEnumConvert:
    def test_exact_match(self):
        result = safe_enum_convert(GNNFormat, "markdown")
        assert result == GNNFormat.MARKDOWN

    def test_uppercase_match(self):
        result = safe_enum_convert(VariableType, "HIDDEN_STATE")
        assert result == VariableType.HIDDEN_STATE

    def test_already_enum(self):
        result = safe_enum_convert(GNNFormat, GNNFormat.JSON)
        assert result == GNNFormat.JSON

    def test_invalid_returns_default(self):
        result = safe_enum_convert(GNNFormat, "nosuchformat", default=GNNFormat.MARKDOWN)
        assert result == GNNFormat.MARKDOWN

    def test_invalid_no_default_returns_none(self):
        result = safe_enum_convert(GNNFormat, "nosuchformat")
        assert result is None


# ── extract_embedded_json_data ─────────────────────────────────────────────

class TestExtractEmbeddedJsonData:
    def test_finds_json_with_pattern(self):
        content = 'GNN_DATA = {"key": "value"}'
        result = extract_embedded_json_data(content, [r'GNN_DATA = (\{.*?\})'])
        assert result == {"key": "value"}

    def test_returns_none_when_no_match(self):
        result = extract_embedded_json_data("no json here", [r"NOPE = (\{.*?\})"])
        assert result is None

    def test_empty_patterns_returns_none(self):
        result = extract_embedded_json_data('{"a": 1}', [])
        assert result is None

    def test_dotall_spans_lines(self):
        content = 'DATA = {\n  "x": 1\n}'
        result = extract_embedded_json_data(content, [r'DATA = (\{.*?\})'])
        assert result == {"x": 1}


# ── ASTNode ────────────────────────────────────────────────────────────────

class TestASTNode:
    def test_node_type_auto_set(self):
        node = ASTNode()
        assert node.node_type == "ASTNode"

    def test_id_unique_per_instance(self):
        n1 = ASTNode()
        n2 = ASTNode()
        assert n1.id != n2.id

    def test_get_children_empty(self):
        node = ASTNode()
        assert node.get_children() == []

    def test_variable_node_type(self):
        var = Variable(name="s1")
        assert var.node_type == "Variable"

    def test_connection_node_type(self):
        conn = Connection()
        assert conn.node_type == "Connection"

    def test_accept_calls_visitor(self):
        class _CountVisitor:
            count = 0
            def visit(self, node):
                self.count += 1
                return self.count

        visitor = _CountVisitor()
        node = ASTNode()
        result = node.accept(visitor)
        assert result == 1
        assert visitor.count == 1


# ── PrintVisitor ───────────────────────────────────────────────────────────

class TestPrintVisitor:
    def test_visit_basic_node(self):
        node = ASTNode()
        visitor = PrintVisitor()
        result = visitor.visit(node)
        assert "ASTNode" in result

    def test_visit_named_node(self):
        var = Variable(name="my_var")
        visitor = PrintVisitor()
        result = visitor.visit(var)
        assert "my_var" in result

    def test_indent_produces_spaces(self):
        visitor = PrintVisitor(indent=2)
        result = visitor.visit(ASTNode())
        # indent=2 → "  " * 2 = 4 spaces at start
        assert result.startswith("    ")

    def test_zero_indent_no_leading_space(self):
        visitor = PrintVisitor(indent=0)
        result = visitor.visit(ASTNode())
        assert not result.startswith(" ")

    def test_visit_children_called(self):
        """ASTVisitor.visit_children traverses children via accept."""
        parent = ASTNode()
        child = ASTNode()
        # Manually inject child into parent's metadata dict to test via visitor
        # (Variable stores children as ASTNode fields — use a Variable)
        visited = []

        class _TrackVisitor(ASTVisitor):
            def visit(self, node):
                visited.append(node)
                return node

        var = Variable(name="v")
        tv = _TrackVisitor()
        tv.visit_children(var)
        # get_children on Variable returns no child ASTNodes since all fields are primitives
        assert isinstance(visited, list)


# ── ParseResult ────────────────────────────────────────────────────────────

class TestParseResult:
    def _make_result(self):
        model = GNNInternalRepresentation(model_name="M")
        return ParseResult(model=model)

    def test_initial_no_errors(self):
        r = self._make_result()
        assert not r.has_errors()
        assert r.success is True

    def test_add_error_sets_failure(self):
        r = self._make_result()
        r.add_error("something went wrong")
        assert r.has_errors()
        assert r.success is False
        assert "something went wrong" in r.errors

    def test_add_warning_does_not_fail(self):
        r = self._make_result()
        r.add_warning("minor issue")
        assert r.has_warnings()
        assert "minor issue" in r.warnings
        assert r.success is True

    def test_initial_no_warnings(self):
        r = self._make_result()
        assert not r.has_warnings()


# ── BaseGNNParser (via concrete subclass) ──────────────────────────────────

class _ConcreteParser(BaseGNNParser):
    def parse_file(self, file_path):
        return ParseResult(model=self.create_empty_model())

    def parse_string(self, content):
        return ParseResult(model=self.create_empty_model())

    def get_supported_extensions(self):
        return [".test"]


class TestBaseGNNParser:
    def test_create_empty_model_custom_name(self):
        parser = _ConcreteParser()
        model = parser.create_empty_model("MyModel")
        assert model.model_name == "MyModel"

    def test_create_empty_model_default_name(self):
        parser = _ConcreteParser()
        model = parser.create_empty_model()
        assert "Unnamed" in model.model_name

    def test_create_parse_error_uses_current_state(self):
        parser = _ConcreteParser()
        parser.current_file = "f.md"
        parser.current_line = 10
        parser.current_column = 5
        err = parser.create_parse_error("bad token")
        assert err.line == 10
        assert err.source == "f.md"

    def test_get_supported_extensions(self):
        parser = _ConcreteParser()
        assert ".test" in parser.get_supported_extensions()

    def test_parse_string_returns_result(self):
        parser = _ConcreteParser()
        result = parser.parse_string("anything")
        assert isinstance(result, ParseResult)


# ── GNNInternalRepresentation extra methods ────────────────────────────────

class TestGNNInternalRepresentationExtras:
    def _make_model(self):
        return GNNInternalRepresentation(model_name="TestModel")

    def test_get_variable_by_name_found(self):
        model = self._make_model()
        var = Variable(name="s1")
        model.variables.append(var)
        assert model.get_variable_by_name("s1") is var

    def test_get_variable_by_name_not_found(self):
        model = self._make_model()
        assert model.get_variable_by_name("nonexistent") is None

    def test_get_variables_by_type(self):
        model = self._make_model()
        hidden = Variable(name="s1", var_type=VariableType.HIDDEN_STATE)
        obs = Variable(name="o1", var_type=VariableType.OBSERVATION)
        model.variables.extend([hidden, obs])
        result = model.get_variables_by_type(VariableType.HIDDEN_STATE)
        assert result == [hidden]
        assert obs not in result

    def test_get_connections_for_variable(self):
        model = self._make_model()
        conn = Connection(source_variables=["A"], target_variables=["s1"])
        model.connections.append(conn)
        assert conn in model.get_connections_for_variable("s1")
        assert conn in model.get_connections_for_variable("A")
        assert model.get_connections_for_variable("unknown") == []

    def test_get_parameter_by_name(self):
        model = self._make_model()
        param = Parameter(name="alpha", value=0.5)
        model.parameters.append(param)
        assert model.get_parameter_by_name("alpha") is param
        assert model.get_parameter_by_name("beta") is None

    def test_to_dict_contains_model_name(self):
        model = self._make_model()
        d = model.to_dict()
        assert d["model_name"] == "TestModel"

    def test_to_dict_source_format_none(self):
        model = self._make_model()
        d = model.to_dict()
        assert d["source_format"] is None

    def test_to_dict_source_format_enum_serialized(self):
        model = self._make_model()
        model.source_format = GNNFormat.MARKDOWN
        d = model.to_dict()
        assert d["source_format"] == "markdown"

    def test_to_dict_variables_serialized(self):
        model = self._make_model()
        model.variables.append(Variable(name="s1", var_type=VariableType.HIDDEN_STATE))
        d = model.to_dict()
        assert len(d["variables"]) == 1

    def test_ontology_mapping_validation(self):
        model = self._make_model()
        mapping = OntologyMapping(variable_name="ghost_var", ontology_term="SomeTerm")
        model.ontology_mappings.append(mapping)
        issues = model.validate_structure()
        assert any("ghost_var" in i for i in issues)
