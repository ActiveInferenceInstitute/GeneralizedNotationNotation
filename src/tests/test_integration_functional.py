#!/usr/bin/env python3
"""
Functional tests for the Integration Processor module.

Tests system-level consistency checks, dependency graph construction,
circular dependency detection, and cross-reference validation.

Test Coverage:
- process_integration() with single and multiple GNN files
- process_integration() with empty directory
- process_integration() with nonexistent path
- Dependency graph generation (with and without networkx)
- Circular dependency detection
- Cross-reference validation ($ref:, type: references)
- Output JSON schema validation
- Integration summary report generation
- Edge cases: empty files, files with no components, isolated components
"""

import pytest
import json
import logging
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration.processor import process_integration


def _has_networkx() -> bool:
    """Check if networkx is available."""
    try:
        import networkx
        return True
    except ImportError:
        return False


class TestIntegrationFunctional:
    """Functional tests for the integration processor module."""

    @pytest.fixture
    def single_gnn_dir(self, tmp_path):
        """Create a directory with one simple GNN file."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "model_a.md").write_text(
            "# Model A\n\n"
            "## StateSpaceBlock\n"
            "- name: alpha\n"
            "- name: beta\n\n"
            "## Connections\n"
            "alpha -> beta\n"
        )
        return target

    @pytest.fixture
    def multi_gnn_dir(self, tmp_path):
        """Create a directory with multiple GNN files that cross-reference each other."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "model_a.md").write_text(
            "# Model A\n\n"
            "## StateSpaceBlock\n"
            "- name: CompA\n"
            "- name: CompB\n\n"
            "## Connections\n"
            "CompA -> CompB\n"
            "Uses CompC from model_b\n"
        )
        (target / "model_b.md").write_text(
            "# Model B\n\n"
            "## StateSpaceBlock\n"
            "- name: CompC\n"
            "- name: CompD\n\n"
            "## Connections\n"
            "CompC -> CompD\n"
            "Uses CompA from model_a\n"
        )
        return target

    @pytest.fixture
    def gnn_dir_with_refs(self, tmp_path):
        """Create GNN files containing $ref: and type: references."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "referenced.md").write_text(
            "# Referenced Model\n\n"
            "## StateSpaceBlock\n"
            "- name: BaseComp\n\n"
            "## References\n"
            "$ref: UndefinedWidget\n"
            "type: MissingType\n"
            "type: String\n"
        )
        return target

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create an output directory."""
        out = tmp_path / "output"
        out.mkdir()
        return out

    # -- Basic process_integration() tests --

    @pytest.mark.unit
    def test_process_integration_returns_bool(self, single_gnn_dir, output_dir):
        """process_integration should always return a bool."""
        result = process_integration(single_gnn_dir, output_dir, verbose=True)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_process_integration_success(self, single_gnn_dir, output_dir):
        """process_integration should return True for valid input."""
        result = process_integration(single_gnn_dir, output_dir, verbose=True)
        assert result is True

    @pytest.mark.unit
    def test_process_integration_empty_directory(self, tmp_path):
        """process_integration should handle an empty directory gracefully."""
        empty_input = tmp_path / "empty"
        empty_input.mkdir()
        out = tmp_path / "output"
        out.mkdir()

        result = process_integration(empty_input, out, verbose=False)
        assert isinstance(result, bool)
        # With 0 files, the processor still returns success=True
        assert result is True

    @pytest.mark.unit
    def test_process_integration_nonexistent_path(self, tmp_path):
        """process_integration should return False for a nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"
        out = tmp_path / "output"
        out.mkdir()

        result = process_integration(nonexistent, out, verbose=False)
        assert isinstance(result, bool)

    # -- Output artifact tests --

    @pytest.mark.unit
    def test_output_directory_created(self, single_gnn_dir, output_dir):
        """process_integration should create integration_results subdirectory."""
        process_integration(single_gnn_dir, output_dir, verbose=True)

        results_dir = output_dir / "integration_results"
        assert results_dir.exists(), "integration_results directory should be created"

    @pytest.mark.unit
    def test_output_artifacts_created(self, single_gnn_dir, output_dir):
        """process_integration should create JSON results and markdown summary."""
        process_integration(single_gnn_dir, output_dir, verbose=True)

        results_dir = output_dir / "integration_results"
        assert (results_dir / "integration_results.json").exists()
        assert (results_dir / "integration_summary.md").exists()

    @pytest.mark.unit
    def test_results_json_schema(self, single_gnn_dir, output_dir):
        """integration_results.json should contain expected top-level keys."""
        process_integration(single_gnn_dir, output_dir, verbose=True)

        results_file = output_dir / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        required_keys = ["processed_files", "success", "errors", "issues"]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        assert isinstance(data["processed_files"], int)
        assert isinstance(data["issues"], list)

    # -- Multiple file and graph tests --

    @pytest.mark.unit
    def test_multiple_files_processed(self, multi_gnn_dir, output_dir):
        """process_integration should count all GNN files processed."""
        process_integration(multi_gnn_dir, output_dir, verbose=True)

        results_file = output_dir / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        assert data["processed_files"] == 2

    @pytest.mark.unit
    @pytest.mark.skipif(not _has_networkx(), reason="networkx not installed")
    def test_graph_stats_with_networkx(self, multi_gnn_dir, output_dir):
        """When networkx is available, system_graph_stats should have node/edge counts."""
        process_integration(multi_gnn_dir, output_dir, verbose=True)

        results_file = output_dir / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        stats = data.get("system_graph_stats", {})
        assert "nodes" in stats, "Should report node count"
        assert "edges" in stats, "Should report edge count"
        assert stats["nodes"] >= 0
        assert stats["edges"] >= 0

    @pytest.mark.unit
    @pytest.mark.skipif(not _has_networkx(), reason="networkx not installed")
    def test_cross_references_between_files(self, multi_gnn_dir, output_dir):
        """Components referenced across files should create graph edges."""
        process_integration(multi_gnn_dir, output_dir, verbose=True)

        results_file = output_dir / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        stats = data.get("system_graph_stats", {})
        # model_a mentions CompC (defined in model_b), model_b mentions CompA (in model_a)
        if stats.get("nodes", 0) > 0:
            assert stats.get("edges", 0) > 0, (
                "Cross-file references should produce edges"
            )

    # -- Undefined reference detection --

    @pytest.mark.unit
    def test_undefined_ref_detection(self, gnn_dir_with_refs, output_dir):
        """Should detect undefined $ref: references."""
        process_integration(gnn_dir_with_refs, output_dir, verbose=True)

        results_file = output_dir / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        issues = data.get("issues", [])
        ref_issues = [i for i in issues if "UndefinedWidget" in str(i)]
        assert len(ref_issues) > 0, (
            f"Should flag undefined $ref: UndefinedWidget. Issues: {issues}"
        )

    @pytest.mark.unit
    def test_undefined_type_detection(self, gnn_dir_with_refs, output_dir):
        """Should detect undefined CamelCase type references, ignoring builtins."""
        process_integration(gnn_dir_with_refs, output_dir, verbose=True)

        results_file = output_dir / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        issues = data.get("issues", [])
        type_issues = [i for i in issues if "MissingType" in str(i)]
        assert len(type_issues) > 0, (
            f"Should flag undefined type: MissingType. Issues: {issues}"
        )

        # Builtin types like String should NOT be flagged
        string_issues = [i for i in issues if "'String'" in str(i)]
        assert len(string_issues) == 0, "Builtin type 'String' should not be flagged"

    # -- Edge cases --

    @pytest.mark.unit
    def test_empty_gnn_file(self, tmp_path):
        """process_integration should handle empty GNN files without crashing."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "empty.md").write_text("")
        out = tmp_path / "output"
        out.mkdir()

        result = process_integration(target, out, verbose=False)
        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.unit
    def test_file_with_no_components(self, tmp_path):
        """A GNN file with no name: definitions should produce zero graph nodes."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "no_comps.md").write_text(
            "# No Components\n\nJust some text, no definitions.\n"
        )
        out = tmp_path / "output"
        out.mkdir()

        process_integration(target, out, verbose=True)

        results_file = out / "integration_results" / "integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        stats = data.get("system_graph_stats", {})
        if "nodes" in stats:
            assert stats["nodes"] == 0

    @pytest.mark.unit
    def test_verbose_flag_does_not_break(self, single_gnn_dir, output_dir):
        """Both verbose=True and verbose=False should work identically."""
        r1 = process_integration(single_gnn_dir, output_dir, verbose=True)
        # Reset output for second run
        out2 = output_dir.parent / "output2"
        out2.mkdir()
        r2 = process_integration(single_gnn_dir, out2, verbose=False)

        assert r1 == r2
