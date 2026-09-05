"""Tests for visualization.matrix.extract.collect_visualization_matrices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visualization.matrix.extract import collect_visualization_matrices


class TestCollectVisualizationMatrices:
    def test_extracts_from_parameter_list(self) -> None:
        parsed = {
            "parameters": [
                {"name": "A", "value": [[0.9, 0.1], [0.2, 0.8]]},
                {"name": "bad", "value": "not numeric"},
            ]
        }
        matrices = collect_visualization_matrices(parsed)
        assert list(matrices) == ["A"]
        assert matrices["A"].shape == (2, 2)

    def test_extracts_from_parameter_dict_format(self) -> None:
        parsed = {"parameters": {"B": [[1, 0], [0, 1]]}}
        matrices = collect_visualization_matrices(parsed)
        assert "B" in matrices
        assert matrices["B"].shape == (2, 2)

    def test_falls_back_to_variables(self) -> None:
        parsed = {
            "parameters": [],
            "variables": [{"name": "s", "value": [1, 2, 3]}],
        }
        matrices = collect_visualization_matrices(parsed)
        assert "s" in matrices
        assert matrices["s"].size == 3

    def test_falls_back_to_raw_matrices(self) -> None:
        parsed = {
            "parameters": [],
            "variables": [],
            "matrices": [
                {"name": "C", "data": [[2, 0], [0, 2]]},
                {"name": "D", "data": "garbage"},
                {"no_data_key": True},
            ],
        }
        matrices = collect_visualization_matrices(parsed)
        assert list(matrices) == ["C"]
        assert matrices["C"].shape == (2, 2)

    def test_raw_matrix_default_name(self) -> None:
        parsed = {"matrices": [{"data": [[1.0]]}]}
        matrices = collect_visualization_matrices(parsed)
        assert list(matrices) == ["matrix_0"]

    def test_empty_parsed_data(self) -> None:
        assert collect_visualization_matrices({}) == {}

    def test_parameters_win_over_variables_and_matrices(self) -> None:
        parsed = {
            "parameters": [{"name": "P", "value": [[1.0]]}],
            "variables": [{"name": "V", "value": [[2.0]]}],
            "matrices": [{"name": "M", "data": [[3.0]]}],
        }
        matrices = collect_visualization_matrices(parsed)
        assert list(matrices) == ["P"]
        assert isinstance(matrices["P"], np.ndarray)
