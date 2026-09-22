"""Unit tests for the run_cross_framework_comparison execute-module MCP tool.

The registry-level assertions pin name, schema honesty, and envelope shape;
the execute_tool smoke runs against a faked comparison entry, so no Julia or
backend simulation is needed. Paths must live inside the repository
(``resolve_repo_path`` enforces repo-local boundaries), hence the
repo-root-scoped temporary directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.mcp

from gnn.analysis.rxinfer.cross_framework import (
    FRAMEWORKS,
    FrameworkRun,
)
from gnn.mcp.mcp import MCP

REPO_ROOT = Path(__file__).resolve().parents[2]


def _registry() -> MCP:
    from gnn.execute.mcp import register_tools

    registry = MCP(enable_caching=False, enable_rate_limiting=False)
    register_tools(registry)
    return registry


def _fake_runs() -> list[FrameworkRun]:
    return [
        FrameworkRun("rxinfer", "success", "ok"),
        FrameworkRun("pymdp", "unavailable", "pymdp not installed (uv sync)"),
        FrameworkRun("activeinference_jl", "unavailable", "julia is not on PATH"),
        FrameworkRun("jax", "success", "ok"),
        FrameworkRun("pytorch", "unavailable", "torch not installed (uv sync)"),
        FrameworkRun("numpyro", "execution_failed", "exit code 2, no results"),
    ]


class TestToolRegistration:
    @pytest.mark.unit
    def test_tool_registered_with_honest_schema(self) -> None:
        registry = _registry()
        assert "run_cross_framework_comparison" in registry.tools
        tool = registry.tools["run_cross_framework_comparison"]
        assert callable(tool.func)
        assert tool.description
        assert tool.module.endswith("execute")
        assert tool.category == "execute"
        assert set(tool.schema["properties"]) == {
            "gnn_file_path",
            "output_directory",
            "timeout",
        }
        assert tool.schema["required"] == ["gnn_file_path", "output_directory"]
        assert tool.schema["properties"]["timeout"]["type"] == "integer"


class TestToolEnvelope:
    @pytest.mark.unit
    def test_execute_tool_reports_per_framework_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runs = _fake_runs()
        seen: dict[str, Any] = {}

        def fake_compare(
            gnn_file: Path,
            output_dir: Path,
            timeout: int | None = None,
            runtime: Any = None,
        ) -> tuple[str, list[FrameworkRun]]:
            seen["timeout"] = timeout
            html = Path(output_dir) / "model_comparison.html"
            return str(html), runs

        monkeypatch.setattr(
            "gnn.analysis.rxinfer.cross_framework.compare_with_status",
            fake_compare,
        )
        registry = _registry()

        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_dir = Path(tmp)
            gnn_file = tmp_dir / "model.md"
            gnn_file.write_text("# GNN model\n", encoding="utf-8")
            result = registry.execute_tool(
                "run_cross_framework_comparison",
                {
                    "gnn_file_path": str(gnn_file),
                    "output_directory": str(tmp_dir / "out"),
                    "timeout": 42,
                },
            )

        assert result["success"] is True
        assert seen["timeout"] == 42
        assert result["comparison_html"].endswith("model_comparison.html")
        assert result["frameworks_total"] == len(FRAMEWORKS) == 6
        assert result["frameworks_succeeded"] == 2
        assert [entry["framework"] for entry in result["frameworks"]] == list(
            FRAMEWORKS
        )
        assert result["frameworks"][1]["status"] == "unavailable"
        assert result["frameworks"][1]["detail"] == "pymdp not installed (uv sync)"
        assert "2/6" in result["message"]

    @pytest.mark.unit
    def test_envelope_reports_resolution_errors(self) -> None:
        registry = _registry()
        result = registry.execute_tool(
            "run_cross_framework_comparison",
            {
                "gnn_file_path": "no_such_model_anywhere.md",
                "output_directory": "docs",
            },
        )
        assert result["success"] is False
        assert "error" in result
