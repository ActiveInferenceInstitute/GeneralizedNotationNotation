"""
MCP integration for the schema_validator module.

Exposes the syntax-level GNN validation surface (``gnn.schema_validator``)
through MCP: comprehensive single-file validation (schema, round-trip, and
semantic checks via ``GNNValidator``) and regex-based syntax parsing of a
GNN specification file via ``GNNParser``.

The validator/parser imports happen lazily inside the tool bodies: the
``gnn.schema_validator`` package ``__init__`` pulls in the cross-format
validator stack, which is too heavy for module discovery (which loads every
``mcp.py`` under a 30s timeout).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_tool_envelope


def validate_gnn_comprehensive_mcp(file_path: str) -> Dict[str, Any]:
    """
    Validate a GNN specification file with the full validator pipeline.

    Args:
        file_path: Path to the GNN specification file to validate.

    Returns:
        Dictionary with ``success`` and ``file_path``. On success, the
        validation verdict under ``is_valid`` plus ``errors``, ``warnings``,
        ``suggestions``, ``validation_level``, and ``metadata`` from the
        ``ValidationResult``. Exceptions are converted to
        ``{"success": False, "error": str(e)}`` by the envelope.
    """

    def _build() -> Dict[str, Any]:
        from gnn.schema_validator import validate_gnn_file_comprehensive

        result = validate_gnn_file_comprehensive(file_path)
        return {
            "success": True,
            "file_path": file_path,
            "is_valid": result.is_valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "suggestions": list(result.suggestions),
            "validation_level": result.validation_level.name,
            "metadata": dict(result.metadata),
        }

    return run_tool_envelope(
        _build, wrapper_name="validate_gnn_comprehensive_mcp", logger=logger
    )


def parse_gnn_syntax_mcp(file_path: str) -> Dict[str, Any]:
    """
    Parse a GNN specification file with the regex-based syntax parser.

    Args:
        file_path: Path to the GNN specification file to parse.

    Returns:
        Dictionary with ``success`` and ``file_path``. On success, a summary
        of the parsed structure: ``model_name``, ``version``,
        ``variables`` (name -> data type mapping), ``variable_count``,
        ``connections`` (``source -> target`` strings), and ``parameters``.
        Exceptions are converted to ``{"success": False, "error": str(e)}``
        by the envelope.
    """

    def _build() -> Dict[str, Any]:
        from gnn.schema_validator import GNNParser

        # Basic (non-enhanced) parsing keeps the tool deterministic and
        # independent of the round-trip parsing system's optional deps.
        parser = GNNParser(enhanced_validation=False)
        parsed = parser.parse_file(file_path)
        return {
            "success": True,
            "file_path": file_path,
            "model_name": parsed.model_name,
            "version": parsed.version,
            "variables": {
                name: var.data_type for name, var in parsed.variables.items()
            },
            "variable_count": len(parsed.variables),
            "connections": [
                f"{conn.source} -> {conn.target}" for conn in parsed.connections
            ],
            "parameters": dict(parsed.parameters),
        }

    return run_tool_envelope(
        _build, wrapper_name="parse_gnn_syntax_mcp", logger=logger
    )


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register schema_validator tools with the MCP server."""

    mcp_instance.register_tool(
        name="schema_validator.validate_comprehensive",
        func=validate_gnn_comprehensive_mcp,
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the GNN specification file to validate",
                },
            },
            "required": ["file_path"],
        },
        description=(
            "Validate a GNN specification file with the full validator "
            "pipeline (schema, round-trip, and semantic checks)."
        ),
        module=__package__,
        category="schema_validator",
    )

    mcp_instance.register_tool(
        name="schema_validator.parse_syntax",
        func=parse_gnn_syntax_mcp,
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the GNN specification file to parse",
                },
            },
            "required": ["file_path"],
        },
        description=(
            "Parse a GNN specification file with the regex-based syntax "
            "parser and return its structural summary."
        ),
        module=__package__,
        category="schema_validator",
    )

    logger.info("schema_validator module MCP tools registered (2 tools).")
