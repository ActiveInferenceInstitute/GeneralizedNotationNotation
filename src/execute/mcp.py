"""
MCP integration for the execute module.

Exposes GNN execution tools: pipeline execution driver, single-model
GNN execution, PyMDP simulation runner, dependency checker,
and module introspection through MCP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from . import (
    process_execute,
    execute_gnn_model,
    execute_pymdp_simulation,
    check_dependencies,
)


# ── Domain tools ─────────────────────────────────────────────────────────────


def process_execute_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run GNN execution processing for all GNN files in a directory.

    Discovers all `.md` GNN files in `target_directory`, validates them,
    and executes each model using the configured execution backend.

    Args:
        target_directory: Directory containing GNN files.
        output_directory: Directory to save execution results.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag and processing summary.
    """
    try:
        success = process_execute(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": "Execute processing completed" if success else "Execute processing failed",
        }
    except Exception as e:
        logger.error(f"process_execute_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def execute_gnn_model_mcp(
    gnn_file_path: str,
    output_directory: str,
    steps: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute a single GNN model file and return simulation results.

    Parses the GNN file, constructs the corresponding POMDP/Active Inference
    model, runs `steps` inference iterations, and saves outputs.

    Args:
        gnn_file_path: Path to the `.md` GNN model file.
        output_directory: Directory to save execution artifacts.
        steps: Number of simulation steps to run (default 10).
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag, step count, and output file paths.
    """
    try:
        result = execute_gnn_model(
            gnn_file=Path(gnn_file_path),
            output_dir=Path(output_directory),
            steps=steps,
            verbose=verbose,
        )
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": bool(result), "result": result}
    except Exception as e:
        logger.error(f"execute_gnn_model_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def execute_pymdp_simulation_mcp(
    gnn_file_path: str,
    output_directory: str,
    timesteps: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a PyMDP Active Inference simulation from a GNN model file.

    Converts the GNN specification to pymdp matrices (A, B, C, D),
    instantiates an `Agent`, and runs `timesteps` perception-action cycles.

    Args:
        gnn_file_path: Path to the GNN model file.
        output_directory: Directory to write simulation log and plots.
        timesteps: Number of timesteps to simulate (default 20).
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag, timesteps run, VFE trajectory, and output paths.
    """
    try:
        result = execute_pymdp_simulation(
            gnn_file=Path(gnn_file_path),
            output_dir=Path(output_directory),
            timesteps=timesteps,
            verbose=verbose,
        )
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": bool(result), "result": result}
    except Exception as e:
        logger.error(f"execute_pymdp_simulation_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def check_execute_dependencies_mcp() -> Dict[str, Any]:
    """
    Check which execution backend dependencies are installed.

    Probes for: pymdp, numpy, scipy, jax, torch (via find_spec — no import).
    Also checks for Python version compatibility.

    Returns:
        Dictionary with dependency names, availability flags, and versions where detectable.
    """
    try:
        result = check_dependencies()
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": True, "dependencies": result}
    except Exception as e:
        logger.error(f"check_execute_dependencies_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_execute_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the execute module.

    Includes: module version, execution backends available, PyMDP status,
    error-recovery availability, and supported GNN model types.

    Returns:
        Dictionary with module metadata and feature inventory.
    """
    try:
        import importlib
        mod = importlib.import_module(__package__)
        version  = getattr(mod, "__version__", "unknown")
        features = getattr(mod, "FEATURES", {})
        return {
            "success":  True,
            "module":   __package__,
            "version":  version,
            "features": features,
            "tools": [
                "process_execute",
                "execute_gnn_model",
                "execute_pymdp_simulation",
                "check_execute_dependencies",
                "get_execute_module_info",
            ],
        }
    except Exception as e:
        logger.error(f"get_execute_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ──────────────────────────────────────────────────────────


def register_tools(mcp_instance) -> None:
    """Register execute domain tools with the MCP server."""

    mcp_instance.register_tool(
        "process_execute",
        process_execute_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory with GNN files"},
                "output_directory": {"type": "string", "description": "Execution output directory"},
                "verbose":          {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run GNN execution processing: validate and execute all GNN model files in a directory.",
        module=__package__, category="execute",
    )

    mcp_instance.register_tool(
        "execute_gnn_model",
        execute_gnn_model_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path":    {"type": "string", "description": "Path to the GNN model file (.md)"},
                "output_directory": {"type": "string", "description": "Directory to save execution results"},
                "steps":            {"type": "integer", "default": 10, "description": "Simulation steps"},
                "verbose":          {"type": "boolean", "default": False},
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Execute a single GNN model file: parse, construct POMDP, run N inference steps.",
        module=__package__, category="execute",
    )

    mcp_instance.register_tool(
        "execute_pymdp_simulation",
        execute_pymdp_simulation_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path":    {"type": "string", "description": "Path to the GNN model file"},
                "output_directory": {"type": "string", "description": "Directory for simulation outputs"},
                "timesteps":        {"type": "integer", "default": 20, "description": "Number of simulation timesteps"},
                "verbose":          {"type": "boolean", "default": False},
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Run a PyMDP Active Inference simulation from a GNN model (A/B/C/D matrices → Agent → perception-action loop).",
        module=__package__, category="execute",
    )

    mcp_instance.register_tool(
        "check_execute_dependencies",
        check_execute_dependencies_mcp,
        {},
        "Check which execution backend dependencies (pymdp, numpy, scipy, jax) are installed.",
        module=__package__, category="execute",
    )

    mcp_instance.register_tool(
        "get_execute_module_info",
        get_execute_module_info_mcp,
        {},
        "Return version, feature flags, and API surface of the GNN execute module.",
        module=__package__, category="execute",
    )

    logger.info("execute module MCP tools registered (5 real domain tools).")
