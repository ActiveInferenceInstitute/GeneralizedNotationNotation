"""
MCP registration for GUI module (optional; provides discovery hooks if MCP is active).
"""

from __future__ import annotations

from typing import Any, Dict

from mcp import register_module_tools


def register_gui_tools() -> Dict[str, Any]:
    try:
        registered = bool(register_module_tools("gui"))
        return {
            "status": "SUCCESS" if registered else "FAILED",
            "registered": registered,
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}
