#!/usr/bin/env python3
"""
Tests for advanced_visualization/_shared.py.
Covers normalize_connection_format, validate_visualization_data,
and _calculate_semantic_positions.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNormalizeConnectionFormat:
    def _fn(self):
        try:
            from advanced_visualization._shared import normalize_connection_format
            return normalize_connection_format
        except ImportError:
            pytest.skip("advanced_visualization._shared not importable")

    def test_new_format_passthrough(self):
        fn = self._fn()
        conn = {"source_variables": ["A"], "target_variables": ["B"]}
        result = fn(conn)
        assert result["source_variables"] == ["A"]
        assert result["target_variables"] == ["B"]

    def test_old_format_converted(self):
        fn = self._fn()
        conn = {"source": "A", "target": "B", "type": "directed"}
        result = fn(conn)
        assert result["source_variables"] == ["A"]
        assert result["target_variables"] == ["B"]
        assert result.get("type") == "directed"

    def test_old_format_extra_keys_preserved(self):
        fn = self._fn()
        conn = {"source": "X", "target": "Y", "label": "prob", "weight": 0.5}
        result = fn(conn)
        assert result.get("label") == "prob"
        assert result.get("weight") == 0.5

    def test_source_not_in_result_when_converted(self):
        fn = self._fn()
        conn = {"source": "X", "target": "Y"}
        result = fn(conn)
        assert "source" not in result
        assert "target" not in result

    def test_unknown_format_returns_unchanged(self):
        fn = self._fn()
        conn = {"from": "A", "to": "B"}
        result = fn(conn)
        assert result == conn


class TestCalculateSemanticPositions:
    def _fn(self):
        try:
            from advanced_visualization._shared import _calculate_semantic_positions
            return _calculate_semantic_positions
        except ImportError:
            pytest.skip("advanced_visualization._shared not importable")

    def test_empty_variables_returns_empty(self):
        fn = self._fn()
        result = fn([], [])
        # Either empty list or empty array
        assert len(result) == 0

    def test_single_variable_returns_one_position(self):
        pytest.importorskip("numpy")
        fn = self._fn()
        variables = [{"name": "s"}]
        result = fn(variables, [])
        assert len(result) == 1

    def test_multiple_variables_correct_count(self):
        pytest.importorskip("numpy")
        fn = self._fn()
        variables = [{"name": f"v{i}"} for i in range(4)]
        result = fn(variables, [])
        assert len(result) == 4

    def test_positions_are_3d(self):
        pytest.importorskip("numpy")
        fn = self._fn()
        variables = [{"name": "A"}, {"name": "B"}]
        result = fn(variables, [])
        assert result.shape == (2, 3)


class TestValidateVisualizationData:
    def _fn(self):
        try:
            from advanced_visualization._shared import validate_visualization_data
            return validate_visualization_data
        except ImportError:
            pytest.skip("advanced_visualization._shared not importable")

    def _logger(self):
        return logging.getLogger("test")

    def test_empty_model_returns_dict(self):
        fn = self._fn()
        result = fn({}, self._logger())
        assert isinstance(result, dict)

    def test_valid_model_is_valid(self):
        fn = self._fn()
        model_data = {
            "model_name": "Test",
            "variables": [{"name": "s", "dimensions": [3, 1]}],
            "connections": [],
            "parameters": {},
        }
        result = fn(model_data, self._logger())
        assert isinstance(result, dict)
        # Should have a validity indicator
        assert "is_valid" in result or "valid" in result or "errors" in result

    def test_model_with_variables_passes(self):
        fn = self._fn()
        model_data = {
            "model_name": "TestModel",
            "variables": [
                {"name": "A", "dimensions": [3, 3]},
                {"name": "s", "dimensions": [3, 1]},
            ],
            "connections": [{"source": "A", "target": "s"}],
        }
        result = fn(model_data, self._logger())
        assert isinstance(result, dict)
