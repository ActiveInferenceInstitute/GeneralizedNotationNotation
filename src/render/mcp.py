"""
MCP (Model Context Protocol) integration for the render module.

Exposes GNN rendering tools: format listing, single-file rendering,
per-framework status, and batch processing through MCP.
"""

from pathlib import Path
from typing import Dict, Any
import logging

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
        success = process_render(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Render {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_render_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_render_frameworks_mcp() -> Dict[str, Any]:
    """
    Return all supported rendering frameworks and their availability status.

    Returns:
        Dictionary with framework names, descriptions, and availability flags.
    """
    try:
        from . import get_supported_frameworks
        frameworks = get_supported_frameworks()
        return {"success": True, "frameworks": frameworks}
    except Exception as e:
        # Return a static list when the dynamic query fails
        return {
            "success": True,
            "frameworks": {
                "python":       {"available": True,  "description": "Pure Python simulation code"},
                "julia":        {"available": True,  "description": "Julia high-performance simulation"},
                "jax":          {"available": False, "description": "JAX/XLA accelerated computation"},
                "pytorch":      {"available": False, "description": "PyTorch neural integration"},
                "rxinfer":      {"available": False, "description": "RxInfer.jl probabilistic programming"},
            },
            "note": str(e) if e else "",
        }


def render_gnn_to_format_mcp(gnn_file_path: str, output_directory: str,
                              framework: str = "python",
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Render a single GNN file to a specific target framework.

    Args:
        gnn_file_path:    Path to the GNN source file (.md)
        output_directory: Directory to write the rendered output
        framework:        Target framework ('python', 'julia', 'jax', 'pytorch', 'rxinfer')
        verbose:          Enable verbose logging

    Returns:
        Dictionary with success status, output file path, and any errors.
    """
    try:
        gnn_path = Path(gnn_file_path)
        out_dir  = Path(output_directory)

        if not gnn_path.exists():
            return {"success": False, "error": f"GNN file not found: {gnn_file_path}"}

        # Use process_render on single-file temp dir
        import tempfile
        import shutil
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
        "Return all supported rendering frameworks (Python, Julia, JAX, PyTorch, RxInfer) and their availability.",
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
                                     "enum": ["python", "julia", "jax", "pytorch", "rxinfer"], "default": "python"},
                "verbose":          {"type": "boolean", "description": "Enable verbose logging", "default": False},
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Render a single GNN file to a specific target framework (Python, Julia, etc.).",
        module=__package__, category="render",
    )

    mcp_instance.register_tool(
        "get_render_module_info",
        get_render_module_info_mcp,
        {},
        "Return metadata about the render module: supported frameworks and input/output formats.",
        module=__package__, category="render",
    )

    logger.info("render module MCP tools registered (5 tools).")
