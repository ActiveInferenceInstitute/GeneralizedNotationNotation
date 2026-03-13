"""
MCP integration for the SAPF (Sound and Probabilistic Freespace) / audio synthesis module.

Exposes SAPF audio generation tools: audio synthesis from GNN models,
backend discovery, audio status, and module metadata through MCP.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

from . import process_gnn_to_audio, get_module_info as _get_mod_info


def process_sapf_mcp(target_directory: str, output_directory: str,
                     verbose: bool = False) -> Dict[str, Any]:
    """
    Generate SAPF audio from GNN model files.

    Converts GNN probabilistic model structure into SAPF (SuperCollider Audio Program
    Format) audio synthesis code and produces audio artifacts.

    Args:
        target_directory: Directory containing GNN files to convert to audio
        output_directory: Directory to save generated audio files and SAPF scripts
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and audio generation summary.
    """
    try:
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        success = process_gnn_to_audio(
            target_dir=Path(target_directory),
            output_dir=out_dir,
            verbose=verbose,
        )
        audio_files = list(out_dir.rglob("*.wav")) + list(out_dir.rglob("*.aiff")) + list(out_dir.rglob("*.sc"))
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "audio_files_generated": len(audio_files),
            "message": f"SAPF audio generation {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_sapf_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_sapf_module_info_mcp() -> Dict[str, Any]:
    """
    Return metadata about the SAPF audio module.

    Returns:
        Dictionary with version, capabilities, and supported audio formats.
    """
    try:
        info = _get_mod_info()
        return {"success": True, **info}
    except Exception as e:
        logger.error(f"get_sapf_module_info_mcp error: {e}", exc_info=True)
        return {
            "success": True,
            "module": "sapf",
            "description": "SAPF audio synthesis from GNN Active Inference models",
            "supported_formats": ["wav", "aiff", "sc (SuperCollider)"],
            "note": str(e),
        }


def list_audio_artifacts_mcp(output_directory: str) -> Dict[str, Any]:
    """
    List audio and SAPF script artifacts in an output directory.

    Args:
        output_directory: Directory containing generated audio files

    Returns:
        Dictionary with audio file inventory and format counts.
    """
    try:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {"success": False, "error": f"Directory not found: {output_directory}"}

        artifacts: List[Dict[str, Any]] = []
        for ext in ["*.wav", "*.aiff", "*.mp3", "*.ogg", "*.sc", "*.scd"]:
            for f in sorted(out_dir.rglob(ext)):
                artifacts.append({
                    "name":   f.name,
                    "type":   f.suffix.lstrip("."),
                    "size_bytes": f.stat().st_size,
                    "path":   str(f.relative_to(out_dir)),
                })

        by_type: Dict[str, int] = {}
        for a in artifacts:
            by_type[a["type"]] = by_type.get(a["type"], 0) + 1

        return {
            "success":          True,
            "output_directory": str(out_dir),
            "total_artifacts":  len(artifacts),
            "by_type":          by_type,
            "artifacts":        artifacts,
        }
    except Exception as e:
        logger.error(f"list_audio_artifacts_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def check_audio_backends_mcp() -> Dict[str, Any]:
    """
    Check which audio generation backends are available.

    Returns:
        Dictionary with backend names and availability flags.
    """
    try:
        from . import check_audio_backends
        result = check_audio_backends()
        return {"success": True, "backends": result}
    except Exception as e:
        # Recovery status check
        backends: Dict[str, Dict[str, Any]] = {}
        import shutil
        backends["supercollider"] = {"available": bool(shutil.which("sclang")),
                                     "description": "SuperCollider language for SAPF"}
        backends["csound"]        = {"available": bool(shutil.which("csound")),
                                     "description": "Csound synthesis engine"}
        try:
            import sounddevice  # type: ignore
            backends["sounddevice"] = {"available": True,  "description": "Python sounddevice (playback)"}
        except ImportError:
            backends["sounddevice"] = {"available": False, "description": "Python sounddevice (playback)"}
        return {"success": True, "backends": backends, "note": str(e) if e else ""}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register SAPF audio tools with the MCP server."""

    mcp_instance.register_tool(
        "process_sapf",
        process_sapf_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Directory with GNN files to convert to audio"},
            "output_directory": {"type": "string", "description": "Directory for audio and SAPF script outputs"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Generate SAPF audio from GNN Active Inference models using SuperCollider synthesis.",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "get_sapf_module_info",
        get_sapf_module_info_mcp,
        {},
        "Return metadata about the SAPF audio synthesis module (version, formats, capabilities).",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "list_audio_artifacts",
        list_audio_artifacts_mcp,
        {"type": "object", "properties": {
            "output_directory": {"type": "string", "description": "Directory with generated audio files"},
        }, "required": ["output_directory"]},
        "List audio and SAPF script artifacts in an output directory.",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "check_audio_backends",
        check_audio_backends_mcp,
        {},
        "Check which audio generation backends (SuperCollider, Csound, sounddevice) are available.",
        module=__package__, category="audio",
    )

    logger.info("sapf module MCP tools registered (4 tools).")
