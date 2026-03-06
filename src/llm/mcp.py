"""
MCP integration for the llm module.

Exposes GNN LLM tools: pipeline driver, per-file LLM analysis,
documentation generation, provider listing, and module introspection
through MCP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from . import (
    process_llm,
    analyze_gnn_file_with_llm,
    generate_documentation,
)


# ── Domain tools ─────────────────────────────────────────────────────────────


def process_llm_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run LLM analysis pipeline for all GNN files in a directory.

    Discovers all `.md` GNN files, generates LLM-based summaries and
    documentation for each, and saves structured JSON results.

    Args:
        target_directory: Directory containing GNN model files.
        output_directory: Directory to save LLM analysis outputs.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag and processing summary.
    """
    try:
        success = process_llm(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": "LLM processing completed" if success else "LLM processing failed",
        }
    except Exception as e:
        logger.error(f"process_llm_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def analyze_gnn_with_llm_mcp(
    gnn_file_path: str,
    analysis_type: str = "comprehensive",
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run LLM-based analysis on a single GNN model file.

    Sends the GNN content to an LLM provider (configured via env vars or
    `provider` override) and returns structured analysis: summary, insights,
    complexity assessment, and suggested improvements.

    Args:
        gnn_file_path: Path to the GNN `.md` model file.
        analysis_type: One of "comprehensive", "summary", "complexity", "connections".
        provider: Optional LLM provider override (e.g. "openai", "anthropic", "ollama").

    Returns:
        Dictionary with success flag, analysis type, and LLM output.
    """
    try:
        result = analyze_gnn_file_with_llm(
            gnn_file=Path(gnn_file_path),
            analysis_type=analysis_type,
            provider=provider,
        )
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": True, "analysis": str(result)}
    except Exception as e:
        logger.error(f"analyze_gnn_with_llm_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def generate_llm_documentation_mcp(
    gnn_file_path: str,
    output_path: Optional[str] = None,
    format: str = "markdown",
) -> Dict[str, Any]:
    """
    Generate human-readable documentation for a GNN model using an LLM.

    Produces: title, abstract, StateSpace description, parameter tables,
    mathematical formulations, and usage examples — formatted as Markdown
    or plain text.

    Args:
        gnn_file_path: Path to the GNN `.md` model file.
        output_path: Optional path to write the generated documentation.
        format: Output format — "markdown" (default) or "text".

    Returns:
        Dictionary with success flag and generated documentation string.
    """
    try:
        result = generate_documentation(
            gnn_file=Path(gnn_file_path),
            output_path=Path(output_path) if output_path else None,
            format=format,
        )
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": True, "documentation": str(result)}
    except Exception as e:
        logger.error(f"generate_llm_documentation_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_llm_providers_mcp() -> Dict[str, Any]:
    """
    Return available LLM providers and their configuration status.

    Checks environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OLLAMA_BASE_URL, etc.) and reports which providers are configured.

    Returns:
        Dictionary with provider names and availability/configuration status.
    """
    try:
        import os
        import importlib.util
        providers: Dict[str, Dict[str, Any]] = {}

        # OpenAI
        providers["openai"] = {
            "available": importlib.util.find_spec("openai") is not None,
            "configured": bool(os.getenv("OPENAI_API_KEY")),
        }
        # Anthropic
        providers["anthropic"] = {
            "available": importlib.util.find_spec("anthropic") is not None,
            "configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        }
        # Ollama (local)
        providers["ollama"] = {
            "available": importlib.util.find_spec("ollama") is not None,
            "configured": bool(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
        # Google Gemini
        providers["google"] = {
            "available": importlib.util.find_spec("google.generativeai") is not None,
            "configured": bool(os.getenv("GOOGLE_API_KEY")),
        }

        configured = [p for p, info in providers.items() if info["configured"]]
        return {
            "success":    True,
            "providers":  providers,
            "configured": configured,
            "count":      len(providers),
        }
    except Exception as e:
        logger.error(f"get_llm_providers_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_llm_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the LLM module.

    Includes: module version, supported analysis types, available providers,
    and list of exported public tools.

    Returns:
        Dictionary with module metadata and feature inventory.
    """
    try:
        import importlib
        mod = importlib.import_module(__package__)
        version = getattr(mod, "__version__", "unknown")
        return {
            "success": True,
            "module":  __package__,
            "version": version,
            "analysis_types": ["comprehensive", "summary", "complexity", "connections"],
            "output_formats": ["markdown", "text"],
            "tools": [
                "process_llm",
                "analyze_gnn_with_llm",
                "generate_llm_documentation",
                "get_llm_providers",
                "get_llm_module_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_llm_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ──────────────────────────────────────────────────────────


def initialize_llm_module(mcp_instance) -> None:
    """
    Initialize the LLM module prior to tool registration.
    Loads API keys and configures the default LLM processor.
    """
    try:
        from src.llm import create_processor_from_env
        import asyncio
        logger.info("Initializing LLM module prior to MCP registration...")

        # create_processor_from_env is a coroutine; must be awaited
        try:
            loop = asyncio.get_running_loop()
            # If there's an active loop, wrap in a task
            processor = loop.create_task(create_processor_from_env())
        except RuntimeError:
            processor = asyncio.run(create_processor_from_env())

        if processor:
            logger.info("LLM processor initialized successfully.")
        else:
            logger.warning("LLM processor initialized but returned False/None.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM module: {e}")


def register_tools(mcp_instance) -> None:
    """Register LLM domain tools with the MCP server."""

    mcp_instance.register_tool(
        "process_llm",
        process_llm_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory with GNN files"},
                "output_directory": {"type": "string", "description": "LLM output directory"},
                "verbose":          {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run LLM analysis pipeline for all GNN files in a directory.",
        module=__package__, category="llm",
    )

    mcp_instance.register_tool(
        "analyze_gnn_with_llm",
        analyze_gnn_with_llm_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string", "description": "Path to the GNN model file",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["comprehensive", "summary", "complexity", "connections"],
                    "default": "comprehensive",
                },
                "provider": {
                    "type": "string",
                    "description": "LLM provider override (openai, anthropic, ollama, google)",
                    "nullable": True,
                },
            },
            "required": ["gnn_file_path"],
        },
        "Run LLM-based analysis on a single GNN model file: summary, complexity, connections.",
        module=__package__, category="llm",
    )

    mcp_instance.register_tool(
        "generate_llm_documentation",
        generate_llm_documentation_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {"type": "string", "description": "Path to the GNN model file"},
                "output_path":   {"type": "string", "description": "Optional path to write documentation", "nullable": True},
                "format":        {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            },
            "required": ["gnn_file_path"],
        },
        "Generate human-readable Markdown documentation for a GNN model using an LLM.",
        module=__package__, category="llm",
    )

    mcp_instance.register_tool(
        "get_llm_providers",
        get_llm_providers_mcp,
        {},
        "Return available LLM providers and their API key / configuration status.",
        module=__package__, category="llm",
    )

    mcp_instance.register_tool(
        "get_llm_module_info",
        get_llm_module_info_mcp,
        {},
        "Return version, analysis types, output formats, and tool list of the GNN LLM module.",
        module=__package__, category="llm",
    )

    logger.info("llm module MCP tools registered (5 real domain tools).")
