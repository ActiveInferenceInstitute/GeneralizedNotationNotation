#!/usr/bin/env python3
"""
Functional tests for the Research Processor module.

Tests the rule-based research hypothesis generation that analyzes GNN files
for dimensionality issues, sparse connectivity, and other patterns.

Test Coverage:
- process_research() with valid GNN files
- process_research() with empty directories
- process_research() with nonexistent paths
- Hypothesis generation for high-dimensional models
- Hypothesis generation for sparse connectivity
- Output artifact generation (JSON + markdown report)
- Return type consistency (always bool)
- Edge cases: binary files, empty files, malformed content
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.processor import process_research


class TestResearchFunctional:
    """Functional tests for the research processor module."""

    @pytest.fixture
    def gnn_dir_with_high_dim(self, tmp_path: Any) -> Any:
        """Create a GNN file with high-dimensional matrices (dim > 10)."""
        target = tmp_path / "input"
        target.mkdir()
        gnn_file = target / "high_dim_model.md"
        gnn_file.write_text(
            "# High Dimensional Model\n\n"
            "## StateSpaceBlock\n"
            "A[50,50,type=float]\n"
            "B[20,20,20,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return target

    @pytest.fixture
    def gnn_dir_sparse(self, tmp_path: Any) -> Any:
        """Create a GNN file with many variables but few connections (sparse)."""
        target = tmp_path / "input"
        target.mkdir()
        gnn_file = target / "sparse_model.md"
        # Many name: definitions but only one -> arrow
        gnn_file.write_text(
            "# Sparse Model\n\n"
            "## StateSpaceBlock\n"
            "- name: alpha\n"
            "- name: beta\n"
            "- name: gamma\n"
            "- name: delta\n"
            "- name: epsilon\n\n"
            "## Connections\n"
            "alpha -> beta\n"
        )
        return target

    @pytest.fixture
    def gnn_dir_simple(self, tmp_path: Any) -> Any:
        """Create a simple GNN file with small dimensions and normal connectivity."""
        target = tmp_path / "input"
        target.mkdir()
        gnn_file = target / "simple_model.md"
        gnn_file.write_text(
            "# Simple Model\n\n"
            "## ModelName\nSimpleTest\n\n"
            "## StateSpaceBlock\n"
            "A[3,3,type=float]\n"
            "s[3,1,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return target

    @pytest.fixture
    def output_dir(self, tmp_path: Any) -> Any:
        """Create an output directory."""
        out = tmp_path / "output"
        out.mkdir()
        return out

    @pytest.mark.unit
    def test_process_research_returns_bool(
        self, gnn_dir_simple: Any, output_dir: Any
    ) -> Any:
        """process_research should always return a bool."""
        result = process_research(gnn_dir_simple, output_dir, verbose=True)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    @pytest.mark.unit
    def test_process_research_success_with_valid_files(
        self, gnn_dir_simple: Any, output_dir: Any
    ) -> Any:
        """process_research should return True for a valid directory with GNN files."""
        result = process_research(gnn_dir_simple, output_dir, verbose=True)
        assert result is True

    @pytest.mark.unit
    def test_process_research_empty_directory(self, tmp_path: Any) -> Any:
        """process_research should handle an empty input directory gracefully."""
        empty_input = tmp_path / "empty_input"
        empty_input.mkdir()
        out = tmp_path / "output"
        out.mkdir()

        result = process_research(empty_input, out, verbose=False)
        assert isinstance(result, bool)
        # Should still succeed even with 0 files processed
        assert result is True

    @pytest.mark.unit
    def test_process_research_nonexistent_path(self, tmp_path: Any) -> Any:
        """process_research should return False for a nonexistent target directory."""
        nonexistent = tmp_path / "does_not_exist"
        out = tmp_path / "output"
        out.mkdir()

        result = process_research(nonexistent, out, verbose=False)
        assert result is False

    @pytest.mark.unit
    def test_output_artifacts_created(
        self, gnn_dir_simple: Any, output_dir: Any
    ) -> Any:
        """process_research should create research_results.json and research_report.md."""
        process_research(gnn_dir_simple, output_dir, verbose=True)

        results_json = output_dir / "research_results.json"
        report_md = output_dir / "research_report.md"

        assert results_json.exists(), "research_results.json should be created"
        assert report_md.exists(), "research_report.md should be created"

    @pytest.mark.unit
    def test_results_json_schema(self, gnn_dir_simple: Any, output_dir: Any) -> Any:
        """research_results.json should have the expected schema."""
        process_research(gnn_dir_simple, output_dir, verbose=True)

        results_json = output_dir / "research_results.json"
        with open(results_json) as f:
            data = json.load(f)

        assert "processed_files" in data
        assert "success" in data
        assert "hypotheses_generated" in data
        assert "errors" in data
        assert isinstance(data["processed_files"], int)
        assert isinstance(data["hypotheses_generated"], list)

    @pytest.mark.unit
    def test_high_dimension_triggers_hypothesis(
        self, gnn_dir_with_high_dim: Any, output_dir: Any
    ) -> Any:
        """Files with dimensions > 10 should trigger a dimensionality_reduction hypothesis."""
        process_research(gnn_dir_with_high_dim, output_dir, verbose=True)

        with open(output_dir / "research_results.json") as f:
            data = json.load(f)

        hypotheses = data["hypotheses_generated"]
        assert len(hypotheses) > 0, "Should generate hypotheses for high-dim model"

        all_types = [
            h["type"] for entry in hypotheses for h in entry.get("hypotheses", [])
        ]
        assert "dimensionality_reduction" in all_types, (
            f"Expected dimensionality_reduction hypothesis, got types: {all_types}"
        )

    @pytest.mark.unit
    def test_sparse_connectivity_triggers_hypothesis(
        self, gnn_dir_sparse: Any, output_dir: Any
    ) -> Any:
        """Files with low connection-to-variable ratio should trigger connectivity_enrichment."""
        process_research(gnn_dir_sparse, output_dir, verbose=True)

        with open(output_dir / "research_results.json") as f:
            data = json.load(f)

        hypotheses = data["hypotheses_generated"]
        assert len(hypotheses) > 0, "Should generate hypotheses for sparse model"

        all_types = [
            h["type"] for entry in hypotheses for h in entry.get("hypotheses", [])
        ]
        assert "connectivity_enrichment" in all_types, (
            f"Expected connectivity_enrichment hypothesis, got types: {all_types}"
        )

    @pytest.mark.unit
    def test_multiple_gnn_files(self, tmp_path: Any) -> Any:
        """process_research should handle multiple GNN files in one directory."""
        target = tmp_path / "multi_input"
        target.mkdir()
        out = tmp_path / "output"
        out.mkdir()

        for i in range(3):
            (target / f"model_{i}.md").write_text(
                f"# Model {i}\n## StateSpaceBlock\nA[{(i + 1) * 10},{(i + 1) * 10},type=float]\n"
            )

        result = process_research(target, out, verbose=True)
        assert result is True

        with open(out / "research_results.json") as f:
            data = json.load(f)
        assert data["processed_files"] == 3

    @pytest.mark.unit
    def test_empty_gnn_file(self, tmp_path: Any) -> Any:
        """process_research should handle an empty GNN file without crashing."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "empty.md").write_text("")
        out = tmp_path / "output"
        out.mkdir()

        result = process_research(target, out, verbose=False)
        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.unit
    def test_recursive_unicode_analysis_is_deterministic(self, tmp_path: Any) -> None:
        target = tmp_path / "input" / "nested"
        target.mkdir(parents=True)
        (target / "greek.md").write_text(
            "## StateSpaceBlock\nπ[4,type=categorical]\nA[2,3]\n",
            encoding="utf-8",
        )
        first = tmp_path / "first"
        second = tmp_path / "second"

        assert process_research(tmp_path / "input", first, recursive=True)
        assert process_research(tmp_path / "input", second, recursive=True)

        first_bytes = (first / "research_results.json").read_bytes()
        second_bytes = (second / "research_results.json").read_bytes()
        assert first_bytes == second_bytes
        data = json.loads(first_bytes)
        evidence = data["hypotheses_generated"][0]["analysis_evidence"]
        assert evidence["dimensions"]["π"] == [4]
        assert data["hypotheses_generated"][0]["file"] == "nested/greek.md"
        assert data["claim_scope"] == "prospective_unvalidated_hypotheses"

    @pytest.mark.unit
    def test_recursive_same_named_models_keep_distinct_evidence_and_claim_markers(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "input"
        for branch, family in (("a", "HiddenMarkovModel"), ("b", "ActInfPOMDP")):
            nested = target / branch
            nested.mkdir(parents=True)
            (nested / "model.md").write_text(
                "## GNNSection\n"
                f"{family}\n"
                "## StateSpaceBlock\n"
                "A[2,2]\nB[2,2]\nC[2]\nD[2]\npi[2]\n",
                encoding="utf-8",
            )
        first = tmp_path / "first"
        second = tmp_path / "second"

        assert process_research(target, first, recursive=True)
        assert process_research(target, second, recursive=True)
        first_bytes = (first / "research_results.json").read_bytes()
        assert first_bytes == (second / "research_results.json").read_bytes()
        receipt = json.loads(first_bytes)

        assert set(receipt["model_families_detected"]) == {
            "a/model.md",
            "b/model.md",
        }
        assert {entry["file"] for entry in receipt["hypotheses_generated"]} == {
            "a/model.md",
            "b/model.md",
        }
        for entry in receipt["hypotheses_generated"]:
            for hypothesis in entry["hypotheses"]:
                assert hypothesis["source"] == "rule_based_static_analysis"
                assert hypothesis["claim_scope"] == "prospective_unvalidated_hypothesis"

    @pytest.mark.unit
    def test_invalid_recursive_option_fails_with_structured_receipt(
        self, gnn_dir_simple: Any, tmp_path: Path
    ) -> None:
        output = tmp_path / "invalid_recursive"

        assert process_research(gnn_dir_simple, output, recursive="yes") is False
        receipt = json.loads(
            (output / "research_results.json").read_text(encoding="utf-8")
        )
        assert receipt["processed_files"] == 0
        assert receipt["errors"][0]["error_type"] == "invalid_configuration"


@pytest.mark.unit
def test_committed_hierarchy_exemplar_uses_hierarchical_reasoning_path() -> None:
    from research.processor import (
        count_connections,
        detect_model_family,
        extract_state_space_dims,
        generate_rule_based_hypotheses,
    )

    repo_root = Path(__file__).parents[3]
    content = (
        repo_root / "input/gnn_files/hierarchical/temporal_hierarchy.md"
    ).read_text(encoding="utf-8")
    family = detect_model_family(content)
    dims = extract_state_space_dims(content)
    connections = count_connections(content)
    hypotheses = generate_rule_based_hypotheses(content, family, dims, connections)

    assert family == "hierarchical"
    assert dims["A_level0"] == [3, 4]
    assert connections["total"] > 0
    assert "message_passing" in {hypothesis["type"] for hypothesis in hypotheses}


@pytest.mark.unit
def test_section_parsing_is_case_insensitive_and_rejects_nonpositive_dims() -> None:
    from research.processor import (
        count_connections,
        detect_model_family,
        extract_state_space_dims,
        generate_rule_based_hypotheses,
    )

    content = (
        "## gnnsection\nActInfPOMDP\n"
        "## statespaceblock\n"
        "A[2,2]\nB[2,2,2]\nC[2]\nD[2]\npi[2]\ninvalid[2,0]\n"
        "## connections\ns > o\n"
        "## actinfontologyannotation\ns=HiddenState\n"
        "## initialparameterization\nA={(1,0),(0,1)}\n"
    )
    dims = extract_state_space_dims(content)
    connections = count_connections(content)
    family = detect_model_family(content)
    hypothesis_types = {
        hypothesis["type"]
        for hypothesis in generate_rule_based_hypotheses(
            content, family, dims, connections
        )
    }

    assert family == "pomdp"
    assert dims["A"] == [2, 2]
    assert "invalid" not in dims
    assert connections == {"directed": 1, "undirected": 0, "total": 1}
    assert "ontology_annotation" not in hypothesis_types
    assert "parameterization" not in hypothesis_types


@pytest.mark.unit
def test_llm_hypothesis_validation_rejects_unstructured_claims() -> None:
    from research.processor import _validate_llm_hypotheses

    value = [
        {"type": "Bad Type", "description": "x", "rationale": "y", "priority": "high"},
        {
            "type": "test_precision",
            "description": "Test precision sensitivity",
            "rationale": "The model exposes a precision parameter.",
            "priority": "medium",
            "unsupported": "discard me",
        },
    ]

    assert _validate_llm_hypotheses(value) == [
        {
            "type": "test_precision",
            "description": "Test precision sensitivity",
            "rationale": "The model exposes a precision parameter.",
            "priority": "medium",
            "source": "llm_generated",
            "claim_scope": "prospective_unvalidated_hypothesis",
        }
    ]


@pytest.mark.unit
def test_model_family_fallback_requires_complete_pomdp_structure() -> None:
    from research.processor import detect_model_family

    complete = "\n".join([" A[2,2]", "B[2,2,2]", "C[2]", "D[2]", "π[2]"])
    incomplete = "\n".join(["A[2,2]", "B[2,2,2]", "π[2]"])

    assert detect_model_family(complete) == "pomdp"
    assert detect_model_family(incomplete) == "unknown"
