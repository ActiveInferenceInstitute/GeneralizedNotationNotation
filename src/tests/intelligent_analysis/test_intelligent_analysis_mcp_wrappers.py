#!/usr/bin/env python3
"""
Functional tests for MCP tool wrappers across the gnn-analysis cluster's
intelligent_analysis and ontology modules.

All tests exercise the real implementations.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from intelligent_analysis.mcp import (  # noqa: E402
    get_analysis_capabilities_mcp,
    get_intelligent_analysis_module_info_mcp,
    process_intelligent_analysis_mcp,
)
from ontology.mcp import list_standard_ontology_terms_mcp  # noqa: E402


class TestIntelligentAnalysisMcp:
    """Exercise the intelligent_analysis MCP wrappers."""

    @pytest.mark.unit
    def test_get_capabilities_mcp(self) -> None:
        """Capabilities should include supported analysis types and features."""
        result = get_analysis_capabilities_mcp()
        assert result["success"] is True
        assert "supported_analysis_types" in result
        assert "failure_analysis" in result["supported_analysis_types"]
        assert "features" in result

    @pytest.mark.unit
    def test_get_module_info_mcp(self) -> None:
        """Module info should expose version and tool inventory."""
        result = get_intelligent_analysis_module_info_mcp()
        assert result["success"] is True
        assert result["module"] == "intelligent_analysis"
        assert isinstance(result["version"], str)
        assert len(result["tools"]) == 3

    @pytest.mark.unit
    def test_process_intelligent_analysis_mcp_empty(self, tmp_path: Path) -> None:
        """Processing with a missing summary should not raise bare errors."""
        target = tmp_path / "target"
        target.mkdir()
        out = tmp_path / "out"
        result = process_intelligent_analysis_mcp(str(target), str(out))
        assert "success" in result
        assert "target_directory" in result

    @pytest.mark.unit
    def test_process_with_analysis_types(self, tmp_path: Path) -> None:
        """analysis_types should be parsed into a list and not raise."""
        target = tmp_path / "target2"
        target.mkdir()
        out = tmp_path / "out2"
        result = process_intelligent_analysis_mcp(
            str(target), str(out), analysis_types="failure_analysis,performance_analysis"
        )
        assert "success" in result

    @pytest.mark.unit
    def test_process_nonexistent_target(self, tmp_path: Path) -> None:
        """A nonexistent target should still return a structured response."""
        out = tmp_path / "out3"
        out.mkdir()
        result = process_intelligent_analysis_mcp(
            str(tmp_path / "missing"), str(out)
        )
        assert "success" in result


class TestOntologyMcpSupplement:
    """Additional ontology MCP wrapper coverage."""

    @pytest.mark.unit
    def test_list_terms_includes_descriptions(self) -> None:
        """Standard terms should include a description for each."""
        result = list_standard_ontology_terms_mcp()
        assert result["success"] is True
        for name, desc in result["terms"].items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert desc