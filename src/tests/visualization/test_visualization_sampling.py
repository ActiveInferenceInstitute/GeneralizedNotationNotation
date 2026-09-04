"""Tests for visualization.core.sampling (pure downsampling helpers)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.core.sampling import (
    MATRIX_SAMPLE_LIMIT,
    VARIABLE_SAMPLE_LIMIT,
    sample_parsed_data,
)


def _variables(count: int) -> List[Dict[str, Any]]:
    return [{"name": f"s{i}", "var_type": "hidden_state"} for i in range(count)]


def _connections() -> List[Dict[str, Any]]:
    return [
        {"source_variables": ["s0"], "target_variables": ["s1"]},
        {"source_variables": ["s999"], "target_variables": ["s1"]},
        {"source_variables": ["s998"], "target_variables": ["s997"]},
    ]


class TestSampleParsedData:
    def test_below_limit_is_noop(self) -> None:
        data: Dict[str, Any] = {
            "variables": _variables(10),
            "connections": _connections(),
        }
        assert sample_parsed_data(data) is False
        assert len(data["variables"]) == 10
        assert "_sampling_applied" not in data

    def test_missing_variables_is_noop(self) -> None:
        assert sample_parsed_data({}) is False
        assert sample_parsed_data({"variables": None}) is False

    def test_non_list_variables_is_noop(self) -> None:
        assert sample_parsed_data({"variables": "not-a-list"}) is False

    def test_sampling_truncates_variables_and_filters_connections(self) -> None:
        data: Dict[str, Any] = {
            "variables": _variables(150),
            "connections": _connections(),
        }
        assert sample_parsed_data(data) is True
        assert len(data["variables"]) == VARIABLE_SAMPLE_LIMIT
        # s0/s1 survive; connections touching s997..s999 are filtered out.
        assert len(data["connections"]) == 1
        summary = data["_sampling_applied"]
        assert summary["original_variables"] == 150
        assert summary["sampled_variables"] == VARIABLE_SAMPLE_LIMIT
        assert summary["original_connections"] == 3
        assert summary["sampled_connections"] == 1

    def test_sampling_caps_matrices(self) -> None:
        data: Dict[str, Any] = {
            "variables": _variables(120),
            "matrices": [{"name": f"m{i}", "data": [[1]]} for i in range(10)],
        }
        assert sample_parsed_data(data) is True
        assert len(data["matrices"]) == MATRIX_SAMPLE_LIMIT

    def test_custom_limits(self) -> None:
        data: Dict[str, Any] = {"variables": _variables(6)}
        assert sample_parsed_data(data, variable_limit=4) is True
        assert len(data["variables"]) == 4

    def test_original_data_not_mutated_when_no_sampling(self) -> None:
        variables = _variables(3)
        data: Dict[str, Any] = {"variables": variables}
        sample_parsed_data(data)
        assert data["variables"] is variables
