"""Deterministic tests for llm.analyzer pure extraction and analysis helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.llm.analyzer import (
    calculate_complexity_metrics,
    extract_connections,
    extract_variables,
    identify_patterns,
    perform_semantic_analysis,
    variable_type_counts,
)

pytestmark = pytest.mark.unit

SECTIONED_MODEL = """## ModelName
TestAgent

## StateSpaceBlock
s_f0[3,1]: boolean
o_m0[2,1]: categorical

## Connections
s_f0 > o_m0
s_f0 - s_f1
o_m0 < s_f0
"""

FREEFORM_MODEL = "a -> b\nc → d\ne connects f\n"


class TestExtractConnections:
    def test_section_operators_parsed_with_types(self) -> None:
        conns = extract_connections(SECTIONED_MODEL)
        pairs = [(c["source"], c["target"], c["connection_type"]) for c in conns]
        assert ("s_f0", "o_m0", "directional") in pairs
        assert ("s_f0", "s_f1", "bidirectional") in pairs
        assert ("o_m0", "s_f0", "reverse") in pairs
        assert all(c["line"] > 0 for c in conns)

    def test_fallback_patterns_when_section_absent(self) -> None:
        conns = extract_connections(FREEFORM_MODEL)
        pairs = {(c["source"], c["target"]) for c in conns}
        assert ("a", "b") in pairs
        assert ("c", "d") in pairs
        assert ("e", "f") in pairs

    def test_no_connections_returns_empty(self) -> None:
        assert extract_connections("no links here") == []


class TestExtractVariables:
    def test_finds_typed_variable(self) -> None:
        variables = extract_variables(SECTIONED_MODEL)
        names = {v["name"] for v in variables}
        assert "s_f0" in names
        assert all(v["line"] > 0 for v in variables)
        assert all("definition" in v for v in variables)

    def test_assignment_pattern(self) -> None:
        variables = extract_variables("learning_rate = 0.1")
        assert any(v["name"] == "learning_rate" for v in variables)

    def test_empty_content(self) -> None:
        assert extract_variables("") == []


class TestVariableTypeCounts:
    def test_typed_and_unknown(self) -> None:
        variables = [
            {"definition": "s_f0[3,1]: boolean"},
            {"definition": "o_m0[2,1]: boolean"},
            {"definition": "rate = 0.5"},
        ]
        counts = variable_type_counts(variables)
        assert counts["boolean"] == 2
        assert counts["unknown"] == 1

    def test_empty(self) -> None:
        assert variable_type_counts([]) == {}


class TestSemanticAnalysis:
    def test_counts_match_inputs(self) -> None:
        variables = extract_variables(SECTIONED_MODEL)
        connections = extract_connections(SECTIONED_MODEL)
        analysis = perform_semantic_analysis(SECTIONED_MODEL, variables, connections)
        assert analysis["variable_count"] == len(variables)
        assert analysis["connection_count"] == len(connections)
        assert analysis["complexity_score"] == len(variables) + len(connections)
        assert sum(analysis["variable_types"].values()) == len(variables)


class TestComplexityMetrics:
    def test_formulas(self) -> None:
        metrics = calculate_complexity_metrics(
            [{"n": i} for i in range(4)], [{"n": i} for i in range(6)]
        )
        assert metrics["total_elements"] == 10
        assert metrics["density"] == 6 / 4
        assert metrics["cyclomatic_complexity"] == 6 - 4 + 2

    def test_zero_variables_density_guarded(self) -> None:
        metrics = calculate_complexity_metrics([], [{"n": 1}])
        assert metrics["density"] == 1  # max(len(vars), 1) divisor


class TestIdentifyPatterns:
    def test_anti_patterns_for_empty_model(self) -> None:
        patterns = identify_patterns("", [], [])
        assert "No variables defined" in patterns["anti_patterns"]
        assert "No connections defined" in patterns["anti_patterns"]

    def test_suggestions_for_large_models(self) -> None:
        variables = [{"n": i} for i in range(25)]
        connections = [{"n": i} for i in range(60)]
        patterns = identify_patterns("content", variables, connections)
        assert "Consider breaking down into smaller modules" in patterns["suggestions"]
        assert "Consider simplifying the model structure" in patterns["suggestions"]
        assert "High variable count - complex model" in patterns["patterns"]
