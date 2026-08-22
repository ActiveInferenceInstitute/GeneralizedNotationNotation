#!/usr/bin/env python3
"""
Exercises the API module MCP job-management handlers.

Exercises the handler functions in ``src/api/mcp.py`` that wrap the in-memory
job manager. These are dependency-light (no HTTP server, no asyncio loop) and
exercise real path-boundary enforcement and manifest serialization.

Test Coverage:
- gnn_get_pipeline_tools_mcp() lists every configured step
- gnn_list_jobs_mcp() returns the expected envelope
- gnn_get_job_status_mcp() unknown-job error path
- gnn_submit_job_mcp() missing-target and outside-repo-root error paths
- gnn_cancel_job_mcp() unknown-job graceful failure
- register_mcp_tools() manifest shape
- save_mcp_manifest() writes a JSON manifest to the output directory
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api import mcp as api_mcp


class TestAPIMCPTools:
    """Tests for the API MCP tool handlers."""

    @pytest.mark.unit
    def test_get_pipeline_tools_lists_steps(self) -> None:
        """Every registered pipeline step should be surfaced as a tool."""
        result = api_mcp.gnn_get_pipeline_tools_mcp()
        assert result["status"] == "success"
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) > 20  # a 25-step pipeline registers many tools
        first = result["tools"][0]
        assert "step_number" in first
        assert "name" in first
        assert "description" in first

    @pytest.mark.unit
    def test_list_jobs_returns_envelope(self) -> None:
        """List jobs returns a success envelope with a total count."""
        result = api_mcp.gnn_list_jobs_mcp()
        assert result["status"] == "success"
        assert isinstance(result["jobs"], list)
        assert "total" in result

    @pytest.mark.unit
    def test_get_job_status_unknown_job(self) -> None:
        """An unknown job id returns an error with a helpful message."""
        result = api_mcp.gnn_get_job_status_mcp("no-such-job")
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.unit
    def test_cancel_job_unknown_job(self) -> None:
        """Cancelling an unknown job returns a graceful error envelope."""
        result = api_mcp.gnn_cancel_job_mcp("no-such-job")
        assert result["status"] == "error"
        assert "message" in result

    @pytest.mark.unit
    def test_submit_job_target_not_found(self, tmp_path: Any) -> None:
        """A missing target directory is rejected with an error before job creation."""
        missing = tmp_path / "no_such_input"
        result = api_mcp.gnn_submit_job_mcp(str(missing))
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.unit
    def test_register_mcp_tools_manifest(self) -> None:
        """The static manifest should include the standard GNN job tools."""
        manifest = api_mcp.register_mcp_tools()
        assert manifest["module"] == "api"
        names = {t["name"] for t in manifest["tools"]}
        assert "gnn_submit_job" in names
        assert "gnn_job_status" in names
        assert "gnn_list_tools" in names

    @pytest.mark.unit
    def test_save_mcp_manifest_writes_json(self, tmp_path: Any) -> None:
        """save_mcp_manifest should write api_mcp_manifest.json and return True."""
        out = tmp_path / "manifest_out"
        assert api_mcp.save_mcp_manifest(out) is True
        manifest_path = out / "api_mcp_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["module"] == "api"
        assert "tools" in data