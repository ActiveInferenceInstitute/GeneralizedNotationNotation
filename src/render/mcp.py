"""
MCP (Model Context Protocol) integration for the render module.

Exposes GNN rendering tools: format listing, single-file rendering,
per-framework status, and batch processing through MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from . import process_render


def process_render_mcp(target_directory: str, output_directory: str,
                       verbose: bool = False) -> Dict[str, Any]:
    """
    Render GNN models in a directory to all supported code formats.

    Args:
        target_directory: Directory containing GNN files to render
        output_directory: Directory to write rendered outputs
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and render summary.
    """
    try:
        raw = process_render(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        # Phase 1.1 contract: process_render may return bool OR int (0/1/2).
        # Coerce to MCP bool envelope; surface "skipped" separately.
        if isinstance(raw, bool):
            success = raw
            skipped = False
        else:  # int
            success = raw in (0, 2)
            skipped = raw == 2
        if skipped:
            message = "Render skipped (no GNN files found)"
        else:
            message = f"Render {'completed successfully' if success else 'completed with issues'}"
        return {
            "success": success,
            "skipped": skipped,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": message,
        }
    except Exception as e:
        logger.error(f"process_render_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# Canonical framework descriptions used by both branches of
# ``list_render_frameworks_mcp`` so success/fallback share one shape.
_FRAMEWORK_DESCRIPTIONS: Dict[str, str] = {
    "pymdp":              "PyMDP Active Inference simulation (Python)",
    "rxinfer":            "RxInfer.jl probabilistic programming (Julia)",
    "activeinference_jl": "ActiveInference.jl simulation (Julia)",
    "jax":                "JAX/XLA accelerated computation (Python)",
    "discopy":            "DisCoPy string diagrams (Python)",
    "pytorch":            "PyTorch neural integration (Python)",
    "numpyro":            "NumPyro probabilistic programming (Python)",
    "stan":               "Stan probabilistic programming (Stan)",
}


def list_render_frameworks_mcp() -> Dict[str, Any]:
    """
    Return all supported rendering frameworks and their availability status.

    Shape:
        ``{"success": bool, "frameworks": {name: {"available": bool, "description": str}, ...}}``

    The fallback branch preserves this shape and sets every framework to
    ``available: False`` so consumers never need to branch on the result type.
    """
    frameworks: Dict[str, Dict[str, Any]] = {
        name: {"available": False, "description": desc}
        for name, desc in _FRAMEWORK_DESCRIPTIONS.items()
    }
    try:
        from . import get_supported_frameworks
        for name in get_supported_frameworks():
            entry = frameworks.setdefault(
                name, {"available": False, "description": name}
            )
            entry["available"] = True
        return {"success": True, "frameworks": frameworks}
    except Exception as e:
        return {
            "success": False,
            "frameworks": frameworks,
            "error": str(e),
        }


def render_gnn_to_format_mcp(gnn_file_path: str, output_directory: str,
                              framework: str = "pymdp",
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Render a single GNN file to a specific target framework.

    Args:
        gnn_file_path:    Path to the GNN source file (.md)
        output_directory: Directory to write the rendered output
        framework:        Target framework name (best-effort hint; see note below)
        verbose:          Enable verbose logging

    Returns:
        Dictionary with success status, output file path, and any errors.
    """
    try:
        gnn_path = Path(gnn_file_path)
        out_dir  = Path(output_directory)

        if not gnn_path.exists():
            return {"success": False, "error": f"GNN file not found: {gnn_file_path}"}

        # Use process_render on a single-file temp dir.
        #
        # NOTE: At present, this MCP tool does not filter to a single framework.
        # It runs Step 11 rendering and returns any artifacts written under the
        # requested output directory. The `framework` parameter is returned for
        # caller context and may be used for filtering in a future revision.
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_in = Path(tmp) / "input"
            tmp_in.mkdir()
            shutil.copy2(gnn_path, tmp_in / gnn_path.name)
            success = process_render(target_dir=tmp_in, output_dir=out_dir, verbose=verbose)

        output_files = list(out_dir.rglob(f"*{gnn_path.stem}*"))
        return {
            "success": success,
            "gnn_file": str(gnn_path),
            "framework": framework,
            "output_directory": str(out_dir),
            "output_files": [str(f) for f in output_files],
            "message": f"Rendering to {framework} {'succeeded' if success else 'failed'}",
        }
    except Exception as e:
        logger.error(f"render_gnn_to_format_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_render_module_info_mcp() -> Dict[str, Any]:
    """
    Return metadata about the render module capabilities.

    Returns:
        Dictionary with version, supported frameworks, and supported input formats.
    """
    try:
        from . import get_supported_frameworks
        frameworks = get_supported_frameworks()
    except Exception:
        frameworks = {}
    return {
        "success": True,
        "module": __package__,
        "frameworks": frameworks,
        "input_formats": ["markdown", "gnn"],
        "output_formats": ["python", "julia", "julia_rxinfer"],
    }


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register render tools with the MCP server."""

    mcp_instance.register_tool(
        "process_render",
        process_render_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing GNN files"},
                "output_directory": {"type": "string", "description": "Directory to write rendered outputs"},
                "verbose":          {"type": "boolean", "description": "Enable verbose logging", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Render GNN models in a directory to all supported code frameworks.",
        module=__package__, category="render",
    )

    mcp_instance.register_tool(
        "list_render_frameworks",
        list_render_frameworks_mcp,
        {},
        "Return supported render framework names and availability (best effort).",
        module=__package__, category="render",
    )

    mcp_instance.register_tool(
        "render_gnn_to_format",
        render_gnn_to_format_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path":    {"type": "string", "description": "Path to the GNN source file (.md)"},
                "output_directory": {"type": "string", "description": "Directory for rendered output"},
                "framework":        {"type": "string", "description": "Target framework",
                                     "enum": ["pymdp", "rxinfer", "activeinference_jl", "jax", "discopy", "pytorch", "numpyro", "stan"],
                                     "default": "pymdp"},
                "verbose":          {"type": "boolean", "description": "Enable verbose logging", "default": False},
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Render a single GNN file (runs Step 11 render; does not currently filter to one framework).",
        module=__package__, category="render",
    )

    mcp_instance.register_tool(
        "get_render_module_info",
        get_render_module_info_mcp,
        {},
        "Return metadata about the render module: supported frameworks and input/output formats.",
        module=__package__, category="render",
    )

    logger.info("render module MCP tools registered (4 tools).")
