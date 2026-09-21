"""
MCP integration for the headless POMDP extraction module.

Exposes the stdlib-only, headless POMDP extractor (``gnn.extract``) through
MCP: a single ``extract_pomdp`` tool that takes a GNN specification file and
returns the extracted POMDP state space as a versioned (1.0.0) JSON payload.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_tool_envelope

EXTRACT_SCHEMA_VERSION = "1.0.0"


def extract_pomdp_mcp(
    file_path: str,
    strict_validation: bool = True,
    on_error: str = "lenient",
    compact: bool = False,
) -> Dict[str, Any]:
    """
    Extract a POMDP state space from a GNN specification file.

    Args:
        file_path: Path to a GNN specification file containing a POMDP.
        strict_validation: Enable strict validation in the extractor.
        on_error: Extractor error mode — ``"lenient"`` (default), ``"raise"``,
            or ``"collect"``. Passed through to the extractor unchanged.
        compact: Emit compact JSON (no indentation) from the extractor
            instead of the default pretty-printed form.

    Returns:
        Dictionary with ``success`` and ``file_path``. On success, the
        extracted POMDP state space under ``pomdp`` plus
        ``schema_version`` (1.0.0). On failure, the extractor's error
        envelope (``status``/``error`` keys) merged into the result with
        ``success`` set to ``False``. Exceptions inside the builder are
        converted to ``{"success": False, "error": str(e)}`` by the
        envelope.
    """

    def _build() -> Dict[str, Any]:
        from gnn.extract import extract_to_json

        payload: Dict[str, Any] = json.loads(
            extract_to_json(
                file_path,
                strict_validation=strict_validation,
                on_error=on_error,
                compact=compact,
            )
        )
        if payload.get("status") == "error":
            return {"success": False, "file_path": file_path, **payload}
        return {
            "success": True,
            "file_path": file_path,
            "schema_version": EXTRACT_SCHEMA_VERSION,
            "pomdp": payload,
        }

    return run_tool_envelope(_build, wrapper_name="extract_pomdp_mcp", logger=logger)


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register POMDP extraction tools with the MCP server."""

    mcp_instance.register_tool(
        "extract_pomdp",
        extract_pomdp_mcp,
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to a GNN specification file containing a POMDP",
                },
                "strict_validation": {
                    "type": "boolean",
                    "description": "Enable strict validation in the extractor",
                    "default": True,
                },
                "on_error": {
                    "type": "string",
                    "description": "Extractor error mode",
                    "enum": ["lenient", "raise", "collect"],
                    "default": "lenient",
                },
                "compact": {
                    "type": "boolean",
                    "description": "Emit compact JSON from the extractor",
                    "default": False,
                },
            },
            "required": ["file_path"],
        },
        "Extract the POMDP state space from a GNN specification file as a "
        "versioned JSON payload.",
        module=__package__,
        category="extract",
    )

    logger.info("extract module MCP tools registered (1 tool).")
