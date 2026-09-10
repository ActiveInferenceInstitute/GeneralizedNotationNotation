#!/usr/bin/env python3
"""
Regression tests for the time-indexed variable-name grammar (S2-8).

'+' in variable names must parse in StateSpaceBlock declarations and
Connections (e.g. ``B_t>s_t+1``) instead of being rejected as unparseable
or misread as arithmetic, and semantic validation must resolve ``s_t+1``
against its declaration in
input/gnn_files/discrete/time_varying_dynamics.md.
"""

from pathlib import Path
from typing import Any

from gnn.types import ValidationLevel

EXEMPLAR = (
    Path(__file__).resolve().parents[2]
    / "input/gnn_files/discrete/time_varying_dynamics.md"
)


class TestTimeIndexedVariableNames:
    """'+' must be part of the variable-name grammar, end to end."""

    def test_schema_parser_connection_target(self) -> Any:
        from gnn.schema import parse_connections

        edges, errors = parse_connections("## Connections\nB_t>s_t+1\n")
        assert errors == []
        assert len(edges) == 1
        assert edges[0].source == "B_t"
        assert edges[0].target == "s_t+1"
        assert edges[0].directed is True

    def test_schema_parser_connection_source(self) -> Any:
        from gnn.schema import parse_connections

        edges, errors = parse_connections("## Connections\ns_t+1>s_t+2\n")
        assert errors == []
        assert len(edges) == 1
        assert edges[0].source == "s_t+1"
        assert edges[0].target == "s_t+2"

    def test_schema_parser_plus_resolves_declared_name(self) -> Any:
        from gnn.schema import parse_connections

        edges, errors = parse_connections(
            "## Connections\nB_t>s_t+1\n",
            known_variables={"B_t", "s_t+1"},
        )
        assert errors == []
        assert len(edges) == 1

    def test_schema_parser_plus_not_arithmetic(self) -> Any:
        """The '+' binds to the name; the target is not split as an expression."""
        from gnn.schema import parse_connections

        edges, _ = parse_connections("## Connections\nA>B_t+C_t\n")
        assert len(edges) == 1
        assert edges[0].target == "B_t+C_t"

    def test_schema_parser_state_space_declares_s_t_plus_1(self) -> Any:
        from gnn.schema import parse_state_space

        variables, errors = parse_state_space(
            "## StateSpaceBlock\n"
            "s_t[3,1,type=float]   # Hidden state at time t\n"
            "s_t+1[3,1,type=float] # Hidden state at time t+1\n"
        )
        assert errors == []
        assert [v.name for v in variables] == ["s_t", "s_t+1"]

    def test_syntax_parser_variable_pattern(self) -> Any:
        from gnn.schema_validator.syntax import GNNParser

        parsed = GNNParser(enhanced_validation=False).parse_content(
            "## StateSpaceBlock\n\ns_t+1[3,1,type=float]  # next state\n"
        )
        assert "s_t+1" in parsed.variables

    def test_syntax_parser_connection_target(self) -> Any:
        from gnn.schema_validator.syntax import GNNParser

        parsed = GNNParser(enhanced_validation=False).parse_content(
            "## StateSpaceBlock\n\n"
            "B_t[3,3,2,type=float]\n"
            "s_t[3,1,type=float]\n"
            "s_t+1[3,1,type=float]\n"
            "\n## Connections\n\nB_t>s_t+1\n"
        )
        assert len(parsed.connections) == 1
        assert parsed.connections[0].source == "B_t"
        assert parsed.connections[0].target == "s_t+1"

    def test_syntax_parser_ontology_mapping(self) -> Any:
        from gnn.schema_validator.syntax import GNNParser

        parsed = GNNParser(enhanced_validation=False).parse_content(
            "## ActInfOntologyAnnotation\n\ns_t+1=HiddenState\n"
        )
        assert parsed.ontology_mappings.get("s_t+1") == "HiddenState"

    def test_exemplar_time_varying_dynamics_is_valid(self) -> Any:
        """Parse + validate the canonical exemplar end to end."""
        from gnn.schema_validator.validator import GNNValidator

        result = GNNValidator().validate_file(
            EXEMPLAR, validation_level=ValidationLevel.STANDARD
        )
        assert result.is_valid, result.errors
