#!/usr/bin/env python3
"""Coverage for the standalone MCP tool modules in export/, model_registry/, and cli/.

These three ``*_mcp.py`` modules (``src/export/mcp.py``, ``src/model_registry/mcp.py``,
``src/cli/mcp.py``) define MCP tool handlers plus a ``register_tools``/``register``
registration entry point, but were previously never imported by any test
(coverage: "module never imported"). These tests exercise each handler and the
registration path directly so the documented MCP surfaces are pinned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeRegistry:
    """A minimal stand-in for the MCP registry (register_tool + execute)."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_tool(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.tools[name] = (args, kwargs)

    def has(self, name: str) -> bool:
        return name in self.tools


class _FakeMCP:
    """A minimal MCP-instance stand-in used by register_tools(mcp_instance)."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_tool(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.tools[name] = (args, kwargs)

    def has(self, name: str) -> bool:
        return name in self.tools


# ---------------------------------------------------------------------------
# export/mcp.py
# ---------------------------------------------------------------------------


def test_export_mcp_list_formats() -> None:
    from export.mcp import list_export_formats_mcp

    result = list_export_formats_mcp()
    assert result["success"] is True
    assert "json" in result["formats"]
    assert "xml" in result["formats"]


def test_export_mcp_validate_format() -> None:
    from export.mcp import validate_export_format_mcp

    ok = validate_export_format_mcp("json")
    assert ok["success"] is True
    assert ok["format"] == "json"
    assert ok["is_valid"] is True

    bad = validate_export_format_mcp("not_a_real_format_xyz")
    assert bad["success"] is True
    assert bad["is_valid"] is False


def test_export_mcp_process_directory(tmp_path: Path) -> None:
    from export.mcp import process_export_mcp

    target = tmp_path / "input"
    target.mkdir()
    (target / "model.md").write_text("## ModelName\nDemo\n", encoding="utf-8")
    out = tmp_path / "out"

    result = process_export_mcp(str(target), str(out))
    # Missing GNN processing results -> not a hard crash; a structured result.
    assert isinstance(result, dict)
    assert "success" in result


def test_export_mcp_register_tools() -> None:
    from export.mcp import register_tools

    mcp = _FakeMCP()
    register_tools(mcp)
    assert mcp.has("process_export")
    assert mcp.has("list_export_formats")
    assert mcp.has("validate_export_format")
    assert mcp.has("export_single_gnn_file")


# ---------------------------------------------------------------------------
# model_registry/mcp.py
# ---------------------------------------------------------------------------


def test_model_registry_mcp_register_and_list(tmp_path: Path) -> None:
    from model_registry.mcp import list_models, register_model

    model_file = tmp_path / "demo_model.md"
    model_file.write_text("## ModelName\nDemo\n", encoding="utf-8")
    registry_path = tmp_path / "registry" / "registry.json"

    result = register_model(str(model_file), str(registry_path))
    # Model registration may require a parsable GNN; either outcome must be a
    # structured result (success or explicit error), not an exception.
    assert isinstance(result, dict)
    assert "status" in result

    listing = list_models(registry_path=str(registry_path))
    # list_models returns a list of model entries.
    assert isinstance(listing, list)


def test_model_registry_mcp_get_and_search(tmp_path: Path) -> None:
    from model_registry.mcp import get_model, search_models

    registry_path = tmp_path / "registry" / "registry.json"
    result = get_model("missing_model", registry_path=str(registry_path))
    assert isinstance(result, dict)

    search = search_models("demo", registry_path=str(registry_path))
    assert isinstance(search, list)


def test_model_registry_mcp_register_tools() -> None:
    from model_registry.mcp import register_tools

    registry = _FakeRegistry()
    ok = register_tools(registry)
    assert ok is True
    assert registry.has("model_registry.register_model")
    assert registry.has("model_registry.list_models")


# ---------------------------------------------------------------------------
# cli/mcp.py
# ---------------------------------------------------------------------------


def test_cli_mcp_health_and_preflight() -> None:
    from cli.mcp import cli_health_check, cli_preflight

    health = cli_health_check()
    assert isinstance(health, dict)

    preflight = cli_preflight()
    assert isinstance(preflight, dict)


def test_cli_mcp_register_tools() -> None:
    from cli.mcp import register_tools

    mcp = _FakeMCP()
    register_tools(mcp)
    assert mcp.has("cli.health")
    assert mcp.has("cli.preflight")
