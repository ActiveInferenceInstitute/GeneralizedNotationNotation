#!/usr/bin/env python3
"""
Regression tests for gnn.multimodel — multi-model splitting and parsing.

Pins the public contract of ``split_models`` and ``parse_multimodel``:
model discovery via ``---`` separators, front-matter stripping, empty/non
inputs without crashing, and per-model structured results.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.multimodel import parse_multimodel, split_models  # noqa: E402


def _single_model_content() -> str:
    """A single GNN model with declared variables and a connection."""
    return "# Test Model\n\n## StateSpaceBlock\nx[2]\ny[1]\n\n## Connections\nx>y\n"


def _multi_model_content() -> str:
    """Two GNN models separated by a ``---`` horizontal rule."""
    return (
        "# Model A\n"
        "## StateSpaceBlock\n"
        "x[2]\n"
        "o[1]\n"
        "## Connections\n"
        "x>o\n"
        "---\n"
        "# Model B\n"
        "## StateSpaceBlock\n"
        "x[2]\n"
        "z[1]\n"
        "## Connections\n"
        "x>z\n"
    )


class TestSplitModels:
    """Coverage for split_models content-block discovery."""

    @pytest.mark.unit
    def test_single_model_returns_one_block(self) -> None:
        content = _single_model_content()
        blocks = split_models(content)
        assert isinstance(blocks, list)
        assert len(blocks) == 1

    @pytest.mark.unit
    def test_multi_model_returns_two_blocks(self) -> None:
        blocks = split_models(_multi_model_content())
        assert len(blocks) == 2
        assert "Model A" in blocks[0]
        assert "Model B" in blocks[1]

    @pytest.mark.unit
    def test_frontmatter_is_stripped(self) -> None:
        content = "---\nmodel_name: demo\ntags: [x]\n---\n" + _single_model_content()
        blocks = split_models(content)
        assert len(blocks) == 1
        assert "model_name" not in blocks[0]

    @pytest.mark.unit
    def test_empty_content_returns_no_blocks(self) -> None:
        assert split_models("") == []
        assert split_models("\n\n\n") == []

    @pytest.mark.unit
    def test_blank_blocks_are_filtered(self) -> None:
        content = "\n---\n---\n" + _single_model_content()
        blocks = split_models(content)
        assert all(b.strip() for b in blocks)
        assert len(blocks) >= 1


class TestParseMultimodel:
    """Coverage of parse_multimodel structured results."""

    @pytest.mark.unit
    def test_single_model_result_shape(self) -> None:
        results = parse_multimodel(_single_model_content(), file_path="/tmp/x.gnn")
        assert len(results) == 1
        model = results[0]
        assert model["model_index"] == 0
        assert model["variable_count"] == 2
        names = [v["name"] for v in model["variables"]]
        assert names == ["x", "y"]
        assert model["connection_count"] == 1
        assert model["connections"][0]["source"] == "x"
        assert model["connections"][0]["target"] == "y"

    @pytest.mark.unit
    def test_multi_model_indices(self) -> None:
        results = parse_multimodel(_multi_model_content(), file_path="multi.gnn")
        assert [m["model_index"] for m in results] == [0, 1]
        assert [m["variable_count"] for m in results] == [2, 2]

    @pytest.mark.unit
    def test_empty_input_no_crash(self) -> None:
        assert parse_multimodel("", file_path="empty.gnn") == []

    @pytest.mark.unit
    def test_malformed_input_reports_errors_without_raising(
        self, isolated_temp_dir: Any
    ) -> None:
        # Garbage body: no recognisable sections, but must not raise.
        results = parse_multimodel("@@@ not a model @@@", file_path="bad.gnn")
        assert isinstance(results, list)
        # A garbage block still yields one model dict with empty variable list.
        assert all(m["variable_count"] == 0 for m in results)

    @pytest.mark.unit
    def test_result_variables_are_serializable_dicts(self) -> None:
        results = parse_multimodel(_single_model_content())
        model = results[0]
        for var in model["variables"]:
            assert set(var.keys()) >= {"name", "dimensions", "dtype", "default"}
