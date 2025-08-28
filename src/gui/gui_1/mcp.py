"""
MCP registration for GUI module (optional; provides discovery hooks if MCP is active).
"""

from __future__ import annotations

from typing import Dict, Any

try:
    from mcp import register_module_tools
except Exception:  # MCP optional
    def register_module_tools(*args, **kwargs):
        return {"status": "noop"}


def register_gui_tools() -> Dict[str, Any]:
    try:
        tools = [
            {
                "name": "gui_status",
                "description": "Query GUI availability and export path",
                "params_schema": {"type": "object", "properties": {}}
            }
        ]
        return register_module_tools("gui", tools)
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


