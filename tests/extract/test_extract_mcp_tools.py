#!/usr/bin/env python3
"""
Exercises the extract module MCP tool handlers.

Covers the ``extract_pomdp_mcp`` wrapper against the canonical POMDP fixture
(happy path), the typed error envelope (missing file), and the MCP
registration surface (tool name, schema, and wrapper wiring).
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.extract import mcp as extract_mcp


class _FakeRegistry:
    """Minimal recording registry matching the MCP ``register_tool`` surface."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    def register_tool(self, *args: Any, **kwargs: Any) -> None:
        self.registered.append(
            {
                "args": args,
                **kwargs,
            }
        )


_POMDP_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "input"
    / "gnn_files"
    / "pomdp_gridworld"
    / "pomdp_gridworld_3x3.md"
)


class TestExtractPomdpMcp:
    """Tests for the extract_pomdp_mcp wrapper."""

    @pytest.mark.unit
    def test_extract_pomdp_happy_path(self) -> None:
        """The canonical POMDP fixture extracts to a successful payload."""
        result = extract_mcp.extract_pomdp_mcp(str(_POMDP_FIXTURE))
        assert result["success"] is True
        assert result["file_path"] == str(_POMDP_FIXTURE)
        assert result["schema_version"] == "1.0.0"
        pomdp = result["pomdp"]
        assert isinstance(pomdp, dict)
        assert "status" not in pomdp
        # Real payload keys, verified against the actual extractor output.
        assert pomdp["num_states"] == 9
        assert pomdp["num_actions"] == 5
        assert isinstance(pomdp["A_matrix"], list)

    @pytest.mark.unit
    def test_extract_pomdp_missing_file_typed_error(self, tmp_path: Any) -> None:
        """A nonexistent file returns the typed error envelope, not an exception."""
        missing = tmp_path / "no_such_model.md"
        result = extract_mcp.extract_pomdp_mcp(str(missing))
        assert result["success"] is False
        assert result["file_path"] == str(missing)
        assert result["status"] == "error"
        error = result["error"]
        assert isinstance(error, dict)
        assert error["code"]
        assert error["message"]

    @pytest.mark.unit
    def test_extract_pomdp_kwargs_passthrough(self) -> None:
        """compact=True still yields a parseable payload with success True."""
        result = extract_mcp.extract_pomdp_mcp(
            str(_POMDP_FIXTURE), compact=True, strict_validation=True
        )
        assert result["success"] is True
        assert isinstance(result["pomdp"], dict)


class TestExtractRegisterTools:
    """Tests for the extract MCP registration surface."""

    @pytest.mark.unit
    def test_register_tools_registers_one_tool(self) -> None:
        registry = _FakeRegistry()
        result = extract_mcp.register_tools(registry)
        assert result is None
        assert len(registry.registered) == 1
        entry = registry.registered[0]
        assert entry["args"][0] == "extract_pomdp"
        assert entry["args"][1] is extract_mcp.extract_pomdp_mcp
        assert entry["module"] == "gnn.extract"
        assert entry["category"] == "extract"
        assert entry["args"][3]

    @pytest.mark.unit
    def test_register_tools_schema_shape(self) -> None:
        registry = _FakeRegistry()
        extract_mcp.register_tools(registry)
        schema = registry.registered[0]["args"][2]
        assert schema["type"] == "object"
        assert set(schema["required"]) == {"file_path"}
        props = schema["properties"]
        assert props["file_path"]["type"] == "string"
        assert props["strict_validation"]["type"] == "boolean"
        assert props["on_error"]["enum"] == ["lenient", "raise", "collect"]
        assert props["compact"]["type"] == "boolean"


def test_module_payload_is_valid_json_round_trip() -> None:
    """The wrapper output stays a plain dict, no JSON re-parsing needed by callers."""
    result = extract_mcp.extract_pomdp_mcp(str(_POMDP_FIXTURE))
    json.dumps(result)  # must be JSON-serializable for the MCP surface
