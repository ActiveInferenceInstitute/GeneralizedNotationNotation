"""
MCP integration for the execute module.

Exposes GNN execution tools: pipeline execution driver, single-model
GNN execution, PyMDP simulation runner, dependency checker,
and module introspection through MCP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from api.path_utils import PathValidationError, resolve_repo_path

from . import (
    check_dependencies,
    execute_simulation_from_gnn,
    process_execute,
)
from .pymdp.execute_pymdp import execute_from_gnn_file as _pymdp_execute_from_gnn_file

_GNN_MODEL_SUFFIXES = {".md", ".json", ".yaml", ".yml"}


def _resolve_gnn_model_path(path_value: str, *, purpose: str) -> Path:
    path = resolve_repo_path(
        path_value,
        purpose=purpose,
        must_exist=True,
        must_be_dir=False,
    )
    if not path.is_file():
        raise PathValidationError(f"{purpose} must be a file: {path_value}")
    if path.suffix.lower() not in _GNN_MODEL_SUFFIXES:
        allowed = ", ".join(sorted(_GNN_MODEL_SUFFIXES))
        raise PathValidationError(
            f"{purpose} must be a GNN source file ({allowed}): {path_value}"
        )
    return path


def _resolve_output_directory(path_value: str, *, purpose: str) -> Path:
    return resolve_repo_path(
        path_value,
        purpose=purpose,
        must_be_dir=True,
        create=True,
    )


def _resolve_render_output_directory(path_value: str) -> Path:
    path = resolve_repo_path(
        path_value,
        purpose="Render output directory",
        must_exist=True,
        must_be_dir=True,
    )
    summary_file = path / "render_processing_summary.json"
    if not summary_file.is_file():
        raise PathValidationError(
            "Render output directory must contain render_processing_summary.json"
        )
    return path


# ── Domain tools ─────────────────────────────────────────────────────────────


def process_execute_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run execution for trusted Step 11 rendered implementations.

    Executes only scripts referenced by the Step 11
    `render_processing_summary.json` in `target_directory`.

    Args:
        target_directory: Repository-local Step 11 render output directory.
        output_directory: Directory to save execution results.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag and processing summary.
    """
    try:
        target_path = _resolve_render_output_directory(target_directory)
        output_path = _resolve_output_directory(
            output_directory,
            purpose="Execution output directory",
        )
        raw = process_execute(
            target_dir=target_path,
            output_dir=output_path,
            verbose=verbose,
            render_output_dir=target_path,
            require_render_summary=True,
        )
        # Phase 1.1 contract: process_execute may return bool OR int (0/1/2).
        # Coerce to MCP bool envelope, surfacing the "skipped" case separately.
        if isinstance(raw, bool):
            success = raw
            skipped = False
        else:  # int
            success = raw in (0, 2)  # 2 = skipped/warnings = not an error
            skipped = raw == 2
        if skipped:
            message = "Execute processing skipped (no work found)"
        else:
            message = (
                "Execute processing completed"
                if success
                else "Execute processing failed"
            )
        return {
            "success": success,
            "skipped": skipped,
            "target_directory": str(target_path),
            "output_directory": str(output_path),
            "message": message,
        }
    except Exception as e:
        logger.error(f"process_execute_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def execute_gnn_model_mcp(
    gnn_file_path: str,
    output_directory: str,
) -> Dict[str, Any]:
    """
    Execute a single GNN model file via the pipeline executor.

    Delegates to ``execute.execute_simulation_from_gnn`` which dispatches to
    ``GNNExecutor`` (PyMDP by default). The timestep count is read from the
    GNN spec's ``Time`` section by the underlying simulator; callers that need
    to override it should edit the model file.

    Args:
        gnn_file_path: Path to the ``.md`` GNN model file.
        output_directory: Directory to save execution artifacts.

    Returns:
        Dictionary with success flag and execution metadata. The underlying
        return value is merged into the top-level dict.
    """
    try:
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        output_path = _resolve_output_directory(
            output_directory,
            purpose="Execution output directory",
        )
        result = execute_simulation_from_gnn(
            gnn_path,
            output_path,
        )
        if isinstance(result, dict):
            merged: dict[str, Any] = {"success": bool(result.get("success", True))}
            merged.update(result)
            return merged
        return {"success": bool(result), "result": result}
    except Exception as e:
        logger.error(f"execute_gnn_model_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def execute_pymdp_simulation_mcp(
    gnn_file_path: str,
    output_directory: str,
) -> Dict[str, Any]:
    """
    Run a PyMDP Active Inference simulation from a GNN model file.

    Parses the GNN file into a spec dict, then calls
    ``execute.pymdp.execute_pymdp.execute_from_gnn_file``, which converts the
    spec to pymdp matrices (A, B, C, D), instantiates a pymdp ``Agent``, and
    runs the perception-action loop for the timesteps declared in the model.

    Args:
        gnn_file_path: Path to the GNN model file.
        output_directory: Directory to write the simulation log and plots.

    Returns:
        Dictionary with success flag and the simulator's results dict merged
        at the top level (keys like ``timesteps_run``, ``output_files``, etc.).
    """
    try:
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        output_path = _resolve_output_directory(
            output_directory,
            purpose="PyMDP output directory",
        )
        success, results = _pymdp_execute_from_gnn_file(
            gnn_path,
            output_path,
            correlation_id="mcp",
        )
        payload: Dict[str, Any] = {"success": bool(success)}
        if isinstance(results, dict):
            payload.update(results)
        else:
            payload["result"] = results
        return payload
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
        version = getattr(mod, "__version__", "unknown")
        features = getattr(mod, "FEATURES", {})
        return {
            "success": True,
            "module": __package__,
            "version": version,
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


def register_tools(mcp_instance: Any) -> None:
    """Register execute domain tools with the MCP server."""

    mcp_instance.register_tool(
        "process_execute",
        process_execute_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Repository-local Step 11 render output directory",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Execution output directory",
                },
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run trusted Step 11 rendered scripts listed in render_processing_summary.json.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "execute_gnn_model",
        execute_gnn_model_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN model file (.md)",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save execution results",
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Execute a single GNN model file via GNNExecutor (PyMDP default); timesteps come from the model's Time section.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "execute_pymdp_simulation",
        execute_pymdp_simulation_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN model file",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory for simulation outputs",
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Run a PyMDP Active Inference simulation from a GNN model (A/B/C/D matrices -> Agent -> perception-action loop).",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "check_execute_dependencies",
        check_execute_dependencies_mcp,
        {},
        "Check which execution backend dependencies (pymdp, numpy, scipy, jax) are installed.",
        module=__package__,
        category="execute",
    )

    mcp_instance.register_tool(
        "get_execute_module_info",
        get_execute_module_info_mcp,
        {},
        "Return version, feature flags, and API surface of the GNN execute module.",
        module=__package__,
        category="execute",
    )

    logger.info("execute module MCP tools registered (5 real domain tools).")
