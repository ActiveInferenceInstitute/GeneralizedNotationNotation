"""
ML Integration MCP Integration

This module exposes ML integration processing tools via MCP.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Import utilities from the ml_integration module
from gnn.utils.mcp.dispatch import run_pipeline_step_mcp, run_tool_envelope

from . import FEATURES, check_ml_frameworks, process_ml_integration

# MCP Tools for ML Integration Module


def process_ml_integration_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Process ML integration for GNN files. Exposed via MCP.

    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save ML integration results
        verbose: Enable verbose output

    Returns:
        Dictionary with operation status and results.
    """
    return run_pipeline_step_mcp(
        process_ml_integration,
        wrapper_name="process_ml_integration_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        label="ML integration processing",
        failure_wording="failed",
    )


def check_ml_frameworks_mcp() -> Dict[str, Any]:
    """
    Check available ML frameworks and their versions.

    Probes for PyTorch, TensorFlow, JAX, scikit-learn, and pymdp using
    importlib.util.find_spec (metadata-only, no heavy import execution).

    Returns:
        Dictionary with framework names, availability flags, and versions where detectable.
    """
    return run_tool_envelope(
        lambda: {
            "success": True,
            "frameworks": check_ml_frameworks(),
            "features": FEATURES,
        },
        wrapper_name="check_ml_frameworks_mcp",
        logger=logger,
    )


def list_ml_integration_targets_mcp() -> Dict[str, Any]:
    """
    Return the list of GNN-compatible ML integration targets.

    Integration targets are downstream tools that can consume GNN model
    outputs: pymdp, JAX, NumPy, graph-neural-network libraries, etc.

    Returns:
        Dictionary with integration target names and their dependency status.
    """
    import importlib.util

    targets: dict[str, Any] = {
        "pymdp": importlib.util.find_spec("pymdp") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "jax": importlib.util.find_spec("jax") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "scikit_learn": importlib.util.find_spec("sklearn") is not None,
        "numpyro": importlib.util.find_spec("numpyro") is not None,
    }
    available = [t for t, v in targets.items() if v]
    return {
        "success": True,
        "targets": targets,
        "available": available,
        "count": len(available),
    }


def get_ml_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the ML integration module.

    Returns:
        Dictionary with module metadata, supported frameworks, and tool inventory.
    """

    def _build() -> Dict[str, Any]:
        import importlib

        mod = importlib.import_module(__package__)
        return {
            "success": True,
            "module": __package__,
            "version": getattr(mod, "__version__", "unknown"),
            "features": FEATURES,
            "tools": [
                "process_ml_integration",
                "check_ml_frameworks",
                "list_ml_integration_targets",
                "get_ml_module_info",
            ],
        }

    return run_tool_envelope(
        _build,
        wrapper_name="get_ml_module_info_mcp",
        logger=logger,
    )


# MCP Registration Function
def register_tools(mcp_instance: Any) -> None:
    """Register ML integration domain tools with the MCP."""

    mcp_instance.register_tool(
        "process_ml_integration",
        process_ml_integration_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory containing GNN files to process",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save ML integration results",
                },
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Process ML integration for GNN files: model training, inference setup, and framework export.",
        module=__package__,
        category="ml_integration",
    )

    mcp_instance.register_tool(
        "check_ml_frameworks",
        check_ml_frameworks_mcp,
        {},
        "Check available ML frameworks (PyTorch, TensorFlow, JAX, scikit-learn) and their versions.",
        module=__package__,
        category="ml_integration",
    )

    mcp_instance.register_tool(
        "list_ml_integration_targets",
        list_ml_integration_targets_mcp,
        {},
        "Return GNN-compatible ML integration targets and their dependency availability.",
        module=__package__,
        category="ml_integration",
    )

    mcp_instance.register_tool(
        "get_ml_module_info",
        get_ml_module_info_mcp,
        {},
        "Return version, feature flags, and tool inventory of the ML integration module.",
        module=__package__,
        category="ml_integration",
    )

    logger.info("ml_integration module MCP tools registered (4 domain tools).")
