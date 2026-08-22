#!/usr/bin/env python3
"""
Functional tests for the MCP tool wrappers in the gnn-analysis cluster
(ontology, intelligent_analysis, report).

All tests exercise the real implementations.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ontology.mcp import (  # noqa: E402
    extract_ontology_annotations_mcp,
    list_standard_ontology_terms_mcp,
    process_ontology_mcp,
    validate_ontology_terms_mcp,
)
from report.mcp import (  # noqa: E402
    generate_report_mcp,
    get_report_module_info_mcp,
    list_report_formats_mcp,
    process_report_mcp,
    read_report_mcp,
)


class TestOntologyMcp:
    """Exercise the ontology MCP wrappers."""

    @pytest.mark.unit
    def test_extract_annotations_parses_mappings(self) -> None:
        """A GNN annotation section should yield validated mappings."""
        content = (
            "# Model\n"
            "## ActInfOntologyAnnotation\n"
            "s=HiddenState\n"
            "o=Observation\n"
            "weird=NotAStandardTerm\n"
            "## SomeOtherSection\n"
            "x=HiddenState\n"
        )
        result = extract_ontology_annotations_mcp(content)
        assert result["success"] is True
        assert result["annotations"]["s"] == "HiddenState"
        assert "weird" in result["unknown_terms"]
        assert result["valid_count"] == 2  # s and o; x is after break

    @pytest.mark.unit
    def test_extract_annotations_no_annotation_section(self) -> None:
        """Content without the annotation section should yield no mappings."""
        result = extract_ontology_annotations_mcp("# No annotations here\n")
        assert result["success"] is True
        assert result["annotations"] == {}

    @pytest.mark.unit
    def test_list_standard_terms_returns_terms(self) -> None:
        """The standard term list should be non-empty and include core terms."""
        result = list_standard_ontology_terms_mcp()
        assert result["success"] is True
        assert "HiddenState" in result["terms"]
        assert result["count"] > 10

    @pytest.mark.unit
    def test_validate_terms_with_string(self) -> None:
        """A comma-separated string should be split and validated."""
        result = validate_ontology_terms_mcp("HiddenState,Observation")
        assert result["success"] is True
        assert result["terms"] == ["HiddenState", "Observation"]

    @pytest.mark.unit
    def test_validate_terms_with_list(self) -> None:
        """A list of terms should be validated directly."""
        result = validate_ontology_terms_mcp(["HiddenState"])
        assert result["success"] is True
        assert result["terms"] == ["HiddenState"]

    @pytest.mark.unit
    def test_process_ontology_mcp_empty_dir(self, tmp_path: Path) -> None:
        """Processing an empty directory should not raise."""
        target = tmp_path / "target"
        target.mkdir()
        out = tmp_path / "out"
        result = process_ontology_mcp(str(target), str(out))
        assert isinstance(result["success"], bool)


class TestReportMcp:
    """Test report module MCP wrappers."""

    @pytest.mark.unit
    def test_generate_report_mcp_creates_summary(self, tmp_path: Path) -> None:
        """Generating a report through MCP should write output and report success."""
        target = tmp_path / "gnn"
        target.mkdir()
        (target / "model.md").write_text("# Model\n## GNNSection\nActInfPOMDP\n")
        out = tmp_path / "out"
        result = generate_report_mcp(str(target), str(out))
        assert result["success"] is True
        assert result["target_directory"] == str(target)

    @pytest.mark.unit
    def test_list_report_formats(self) -> None:
        """list_report_formats_mcp should succeed with a formats list."""
        result = list_report_formats_mcp()
        assert result["success"] is True
        assert "json" in result["formats"]

    @pytest.mark.unit
    def test_read_report_mcp_reads_json(self, tmp_path: Path) -> None:
        """A JSON report file should be read and parsed."""
        report = tmp_path / "report.json"
        report.write_text('{"key": "value"}', encoding="utf-8")
        result = read_report_mcp(str(report))
        assert result["success"] is True
        assert result["format"] == "json"
        assert result["data"] == {"key": "value"}

    @pytest.mark.unit
    def test_read_report_mcp_missing_file(self, tmp_path: Path) -> None:
        """A missing report file should report failure gracefully."""
        result = read_report_mcp(str(tmp_path / "missing.json"))
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.unit
    def test_process_report_mcp(self, tmp_path: Path) -> None:
        """process_report_mcp should run the report pipeline step."""
        target = tmp_path / "gnn"
        target.mkdir()
        (target / "m.md").write_text("# M\n## GNNSection\n")
        out = tmp_path / "out"
        result = process_report_mcp(str(target), str(out))
        assert result["success"] is True
        assert result["target_directory"] == str(target)

    @pytest.mark.unit
    def test_get_report_module_info(self) -> None:
        """get_report_module_info_mcp should expose module metadata."""
        result = get_report_module_info_mcp()
        assert result["success"] is True
        assert "version" in result