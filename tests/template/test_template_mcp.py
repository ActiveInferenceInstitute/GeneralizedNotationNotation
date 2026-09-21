"""Real behavioral tests for the template step MCP integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gnn.template import mcp

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_INDEX_PATH = REPO_ROOT / "src" / "gnn" / "cli" / "template_index.json"


class _FakeRegistry:
    """Minimal recording registry matching the MCP ``register_tool`` surface."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    def register_tool(self, **kwargs: Any) -> None:
        self.registered.append(kwargs)


def test_get_template_info_reports_capabilities() -> None:
    info = mcp.get_template_info()
    assert info["name"] == "Template Step"
    assert info["step_number"] == 0
    assert "File processing" in info["capabilities"]
    assert info["version"] == "1.0.0"


def test_register_tools_registers_four_tools() -> None:
    registry = _FakeRegistry()
    result = mcp.register_tools(registry)
    assert result is True
    names = [r["name"] for r in registry.registered]
    assert names == [
        "template.process_file",
        "template.process_directory",
        "template.get_info",
        "template.pull",
    ]


def test_register_tools_surfaces_parameters() -> None:
    registry = _FakeRegistry()
    mcp.register_tools(registry)
    process_file = next(
        r for r in registry.registered if r["name"] == "template.process_file"
    )
    param_names = [p["name"] for p in process_file["parameters"]]
    assert "file_path" in param_names
    assert "output_dir" in param_names


def test_register_tools_pull_tool_parameters() -> None:
    registry = _FakeRegistry()
    mcp.register_tools(registry)
    pull = next(r for r in registry.registered if r["name"] == "template.pull")
    params = {p["name"]: p for p in pull["parameters"]}
    assert set(params) == {"name", "output_dir", "dry_run", "overwrite"}
    assert params["dry_run"]["default"] is True


def test_process_file_real_input(tmp_path: Path) -> None:
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Sample\n")
    out_dir = tmp_path / "out"
    result = mcp.process_file(str(input_file), str(out_dir))
    assert result["status"] == "success"
    assert result["input_file"] == str(input_file)
    assert "output_file" in result
    # The processed output must actually exist and contain a JSON report.
    assert Path(result["output_file"]).exists()


def test_process_file_missing_input_returns_error(tmp_path: Path) -> None:
    result = mcp.process_file(str(tmp_path / "nope.md"), str(tmp_path / "out"))
    assert result["status"] == "error"
    assert "nope.md" in result["input_file"]


def test_process_file_report_is_valid_json(tmp_path: Path) -> None:
    input_file = tmp_path / "sample2.md"
    input_file.write_text("# B\n")
    out_dir = tmp_path / "out2"
    result = mcp.process_file(str(input_file), str(out_dir))
    report_path = Path(result["report_file"])
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_pull_template_mcp_dry_run_success(tmp_path: Path) -> None:
    record = json.loads(TEMPLATE_INDEX_PATH.read_text(encoding="utf-8"))[0]
    result = mcp.pull_template_mcp(record["name"], str(tmp_path), dry_run=True)
    assert result["success"] is True
    assert result["dry_run"] is True
    assert result["copied"] is False


def test_pull_template_mcp_unknown_name_returns_error() -> None:
    result = mcp.pull_template_mcp("no-such-template")
    assert result["success"] is False
    assert "Unknown template" in result["error"]


def test_pull_template_mcp_collision_returns_error(tmp_path: Path) -> None:
    record = json.loads(TEMPLATE_INDEX_PATH.read_text(encoding="utf-8"))[0]
    destination = tmp_path / record["filename"]
    destination.write_text("# different content\n", encoding="utf-8")
    result = mcp.pull_template_mcp(record["name"], str(tmp_path), dry_run=True)
    assert result["success"] is False
    assert "Destination exists" in result["error"]
