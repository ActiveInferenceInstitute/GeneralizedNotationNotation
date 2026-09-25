#!/usr/bin/env python3
"""Tests for the pipeline MCP artifact/status receipts.

Covers ``list_step_artifacts_mcp`` (registry-complete artifact inventory
with per-step file caps) and the ``memory_receipts`` extension on
``get_pipeline_status``. All filesystem work happens under ``tmp_path``;
the pipeline output root is wired through the documented
``get_pipeline_config`` loader seam (the loader the MCP tools call, so
patching it redirects every tool without touching the repo's output tree).
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from gnn.pipeline.step_registry import STEPS  # noqa: E402


def _step_number(step: Any) -> int:
    """Step number: the numeric prefix of the script stem (registry's own rule)."""
    return int(step.script_stem.split("_")[0])


def _write_canonical_summary(output_root: Path) -> None:
    """Write a canonical pipeline_execution_summary.json with memory receipts.

    Follows the schema documented for ``00_pipeline_summary/``: a
    ``performance_summary`` block with ``peak_memory_mb`` plus per-step
    entries, including one step with explicit null memory values and one
    with the memory keys missing entirely.
    """
    summary_dir = output_root / "00_pipeline_summary"
    summary_dir.mkdir(parents=True)
    summary = {
        "start_time": "2026-09-25T00:00:00+00:00",
        "end_time": "2026-09-25T00:01:00+00:00",
        "overall_status": "SUCCESS_WITH_WARNINGS",
        "total_duration_seconds": 60.0,
        "arguments": {},
        "environment_info": {},
        "performance_summary": {
            "successful_steps": 2,
            "failed_steps": 1,
            "critical_failures": 0,
            "warnings": 1,
            "peak_memory_mb": 512.5,
            "total_steps": 3,
        },
        "steps": [
            {
                "step_number": 0,
                "script_name": "0_template.py",
                "description": "Template initialization",
                "status": "SUCCESS",
                "exit_code": 0,
                "memory_usage_mb": 128.0,
                "peak_memory_mb": 256.5,
                "memory_delta_mb": 12.5,
            },
            {
                "step_number": 1,
                "script_name": "1_setup.py",
                "description": "Environment setup",
                "status": "SUCCESS",
                "exit_code": 0,
                "memory_usage_mb": None,
                "peak_memory_mb": None,
                "memory_delta_mb": None,
            },
            {
                "step_number": 2,
                "script_name": "2_tests.py",
                "description": "Test suite execution",
                "status": "SUCCESS_WITH_WARNINGS",
                "exit_code": 0,
            },
        ],
    }
    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


@pytest.fixture()
def pipeline_output_root(tmp_path: Any, monkeypatch: Any) -> Path:
    """Synthetic pipeline output root wired through the config loader seam.

    Populates artifact directories for two real registry steps — one with
    files (including a nested subdirectory) and one empty — leaving every
    other registry step absent so ``exists`` flags cover both states.
    """
    from gnn.pipeline import config as pipeline_config

    root = tmp_path / "output"
    root.mkdir()
    _write_canonical_summary(root)

    populated = next(step for step in STEPS if _step_number(step) == 0)
    empty = next(step for step in STEPS if _step_number(step) == 1)

    populated_dir = root / populated.output_dir_name
    populated_dir.mkdir()
    (populated_dir / "report.md").write_text("x" * 100, encoding="utf-8")
    nested = populated_dir / "sub"
    nested.mkdir()
    (nested / "data.json").write_text("{}", encoding="utf-8")

    (root / empty.output_dir_name).mkdir()

    monkeypatch.setattr(
        pipeline_config,
        "get_pipeline_config",
        lambda: {"output_dir": str(root)},
    )
    return root


class TestGetPipelineStatusMemoryReceipts:
    """The memory_receipts extension on get_pipeline_status."""

    def test_receipts_carry_canonical_memory_keys(
        self, pipeline_output_root: Path
    ) -> None:
        from gnn.pipeline.mcp import get_pipeline_status

        result = get_pipeline_status(None)
        assert result["success"] is True
        receipts = result["memory_receipts"]
        assert receipts["peak_memory_mb"] == 512.5
        assert len(receipts["steps"]) == 3

        by_number = {step["step_number"]: step for step in receipts["steps"]}
        recorded = by_number[0]
        assert recorded["description"] == "Template initialization"
        assert recorded["memory_usage_mb"] == 128.0
        assert recorded["peak_memory_mb"] == 256.5
        assert recorded["memory_delta_mb"] == 12.5

        # Explicit null memory values stay null.
        assert by_number[1]["memory_usage_mb"] is None
        assert by_number[1]["peak_memory_mb"] is None
        assert by_number[1]["memory_delta_mb"] is None

        # Missing memory keys degrade to None instead of raising KeyError.
        assert by_number[2]["memory_usage_mb"] is None
        assert by_number[2]["peak_memory_mb"] is None
        assert by_number[2]["memory_delta_mb"] is None

        # Pre-existing returned keys remain untouched.
        assert result["output_directory"] == str(pipeline_output_root)
        assert "execution_summary" in result
        assert "recent_logs" in result
        assert "pipeline_config" in result

    def test_absent_summary_degrades_to_empty_receipts(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        from gnn.pipeline import config as pipeline_config
        from gnn.pipeline.mcp import get_pipeline_status

        empty_root = tmp_path / "empty_output"
        empty_root.mkdir()
        monkeypatch.setattr(
            pipeline_config,
            "get_pipeline_config",
            lambda: {"output_dir": str(empty_root)},
        )

        result = get_pipeline_status(None)
        assert result["success"] is True
        assert result["memory_receipts"] == {"peak_memory_mb": None, "steps": []}


class TestListStepArtifacts:
    """The registry-complete artifact inventory tool."""

    def test_zero_arg_call_reports_full_registry(
        self, pipeline_output_root: Path
    ) -> None:
        from gnn.pipeline.mcp import list_step_artifacts_mcp

        result = list_step_artifacts_mcp()
        assert result["success"] is True
        assert result["output_root"] == str(pipeline_output_root)
        assert result["steps_count"] == len(STEPS)
        assert [step["step_number"] for step in result["steps"]] == [
            _step_number(step) for step in STEPS
        ]
        assert result["steps_with_artifacts"] == 1

        by_number = {step["step_number"]: step for step in result["steps"]}

        populated = by_number[0]
        assert populated["exists"] is True
        assert populated["script_stem"] == "0_template"
        assert populated["file_count"] == 2
        assert populated["total_size_bytes"] == 102
        assert populated["truncated"] is False
        assert [f["name"] for f in populated["files"]] == [
            "report.md",
            "sub/data.json",
        ]
        assert populated["files"][0]["size_bytes"] == 100
        assert populated["files"][1]["size_bytes"] == 2

        empty = by_number[1]
        assert empty["exists"] is True
        assert empty["file_count"] == 0
        assert empty["total_size_bytes"] == 0
        assert empty["files"] == []
        assert empty["truncated"] is False

        missing_step = next(step for step in STEPS if _step_number(step) == 2)
        missing = by_number[2]
        assert missing["exists"] is False
        assert missing["output_dir"] == str(
            pipeline_output_root / missing_step.output_dir_name
        )
        assert missing["file_count"] == 0
        assert missing["total_size_bytes"] == 0
        assert missing["files"] == []

    def test_step_number_filter(self, pipeline_output_root: Path) -> None:
        from gnn.pipeline.mcp import list_step_artifacts_mcp

        result = list_step_artifacts_mcp(step_number=0)
        assert result["success"] is True
        assert result["steps_count"] == 1
        assert result["steps"][0]["step_number"] == 0
        assert result["steps"][0]["file_count"] == 2

        # A number outside the registry yields an empty (still successful) view.
        no_match = list_step_artifacts_mcp(step_number=999)
        assert no_match["success"] is True
        assert no_match["steps"] == []

    def test_file_cap_and_truncation_flag(self, pipeline_output_root: Path) -> None:
        from gnn.pipeline.mcp import list_step_artifacts_mcp

        target = next(step for step in STEPS if _step_number(step) == 1)
        step_dir = pipeline_output_root / target.output_dir_name
        for i in range(25):
            (step_dir / f"file_{i:02d}.txt").write_text("y" * (i + 1), encoding="utf-8")

        result = list_step_artifacts_mcp(step_number=1)
        entry = result["steps"][0]
        assert entry["file_count"] == 25
        assert len(entry["files"]) == 20
        assert entry["truncated"] is True
        assert entry["files"][0]["name"] == "file_00.txt"
        assert entry["files"][0]["size_bytes"] == 1
        # Totals cover the uncapped set, not just the returned slice.
        assert entry["total_size_bytes"] == sum(i + 1 for i in range(25))
