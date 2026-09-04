#!/usr/bin/env python3
"""
Exercises the security module MCP tool handlers.

Exercises the real handler functions in ``security/mcp.py`` directly: file
scanning, report reading, check taxonomy, and registered-tool invocation.
These are lightweight, dependency-free integrations (no external LLM or
network), so they run fast and pin real behavior.

Test Coverage:
- process_security_mcp() success + nonexistent-directory error path
- scan_gnn_file_mcp() pattern detection (eval, subprocess) and clean files
- scan_gnn_file_mcp() file-not-found error path
- list_security_checks_mcp() taxonomy shape
- get_security_report_mcp() with a saved report and missing-directory error
- register_tools() registers the expected tool names on a minimal MCP
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from security import mcp as security_mcp
from tests.helpers import MCPTools


class TestSecurityMCPTools:
    """Functional tests for the security MCP tool handlers."""

    def _clean_file(self, tmp_path: Any) -> Path:
        """Create a vulnerability-free GNN-style file."""
        target = tmp_path / "input"
        target.mkdir(exist_ok=True)
        f: Path = target / "clean_model.md"
        f.write_text(
            "# Clean Model\n\n"
            "## ModelName\nCleanModel\n\n"
            "## StateSpaceBlock\n"
            "A[3,3,type=float]\n"
            "s[3,1,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return f

    def _vuln_file(self, tmp_path: Any) -> Path:
        """Create a file with script-injection antipatterns."""
        target = tmp_path / "input"
        target.mkdir(exist_ok=True)
        f: Path = target / "vulnerable_model.md"
        f.write_text(
            "# Vulnerable Model\n\n"
            "## Notes\n"
            "eval(user_input)\n"
            "subprocess.call(['rm', '-rf'])\n"
            "import os\n"
        )
        return f

    @pytest.mark.unit
    def test_scan_clean_file_detects_no_issues(self, tmp_path: Any) -> Any:
        """A clean file should report low risk and no issues."""
        f = self._clean_file(tmp_path)
        result = security_mcp.scan_gnn_file_mcp(str(f))
        assert result["success"] is True
        assert result["issues_found"] == 0
        assert result["risk_level"] == "low"
        assert result["file"] == str(f)
        assert result["total_lines"] > 0

    @pytest.mark.unit
    def test_scan_vulnerable_file_detects_injection(self, tmp_path: Any) -> Any:
        """eval() and subprocess usages should be flagged as high-risk issues."""
        f = self._vuln_file(tmp_path)
        result = security_mcp.scan_gnn_file_mcp(str(f))
        assert result["success"] is True
        assert result["risk_level"] == "high"
        assert result["issues_found"] >= 2

        patterns = {issue["pattern"] for issue in result["issues"]}
        assert "eval(" in patterns
        assert "subprocess" in patterns
        severities = {issue["severity"] for issue in result["issues"]}
        assert severities == {"high"}

    @pytest.mark.unit
    def test_scan_missing_file_returns_error(self, tmp_path: Any) -> Any:
        """A nonexistent path should return success=False with a clear error."""
        missing = tmp_path / "nope.md"
        result = security_mcp.scan_gnn_file_mcp(str(missing))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.unit
    def test_list_security_checks_taxonomy(self) -> Any:
        """The security-check taxonomy should list the documented scans."""
        result = security_mcp.list_security_checks_mcp()
        assert result["success"] is True
        assert "count" in result and result["count"] == len(result["checks"])
        names = set(result["checks"].keys())
        assert "code_injection" in names
        assert "dependency_cve_scan" in names
        assert "hardcoded_credentials" in names
        for meta in result["checks"].values():
            assert "description" in meta and "severity" in meta

    @pytest.mark.unit
    def test_get_security_report_reads_saved_reports(self, tmp_path: Any) -> Any:
        """Reading a directory with a saved *security*.json report should surface it."""
        out = tmp_path / "reports"
        out.mkdir(exist_ok=True)
        (out / "security_validation_report.json").write_text(
            json.dumps({"success": True, "processed_files": 1})
        )
        (out / "unrelated.txt").write_text("hello")
        result = security_mcp.get_security_report_mcp(str(out))
        assert result["success"] is True
        assert result["reports_found"] == 1
        assert result["reports"][0]["file"] == "security_validation_report.json"
        assert result["reports"][0]["data"]["processed_files"] == 1

    @pytest.mark.unit
    def test_get_security_report_missing_directory(self, tmp_path: Any) -> Any:
        """A missing reports directory should yield a clear error."""
        missing = tmp_path / "no_out"
        result = security_mcp.get_security_report_mcp(str(missing))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.unit
    def test_process_security_mcp_runs_on_directory(self, tmp_path: Any) -> Any:
        """The MCP wrapper should run a real scan and report success."""
        f = self._clean_file(tmp_path)
        out = tmp_path / "output"
        result = security_mcp.process_security_mcp(str(f.parent), str(out))
        assert result["success"] is True
        assert result["target_directory"] == str(f.parent)
        assert out.exists()

    @pytest.mark.unit
    def test_process_security_mcp_nonexistent_target(self, tmp_path: Any) -> Any:
        """A nonexistent target directory should not raise — it returns an error dict."""
        missing = tmp_path / "does_not_exist"
        out = tmp_path / "output"
        result = security_mcp.process_security_mcp(str(missing), str(out))
        # The processor returns False for a missing target (no files found)
        # rather than raising; the wrapper surfaces that without an error key.
        assert isinstance(result, dict)
        assert "success" in result
        assert "target_directory" in result

    @pytest.mark.unit
    def test_register_tools_binds_security_tools(self) -> Any:
        """register_tools should bind the four expected tool names."""
        mcp = MCPTools()
        security_mcp.register_tools(mcp)
        expected = {
            "process_security",
            "scan_gnn_file",
            "get_security_report",
            "list_security_checks",
        }
        assert expected.issubset(mcp.tools.keys())
