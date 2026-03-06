"""
MCP integration for the audio module.

Exposes GNN audio generation tools: process_audio pipeline driver,
backend availability check, audio generation options, audio analysis,
content validation, and module introspection through MCP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from . import (
    process_audio,
    check_audio_backends,
    get_audio_generation_options,
    analyze_audio_characteristics,
    validate_audio_content,
    get_module_info,
)


# ── Domain tools ─────────────────────────────────────────────────────────────


def process_audio_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run GNN audio processing pipeline for all GNN files in a directory.

    Converts GNN model files to audio representations using the configured
    audio backend (scipy, soundfile, pedalboard, or pure-Python fallback).

    Args:
        target_directory: Directory containing GNN files to process.
        output_directory: Directory to write audio output files.
        verbose: Enable verbose logging during processing.

    Returns:
        Dictionary with success flag, counts, and output path.
    """
    try:
        success = process_audio(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": "Audio processing completed successfully" if success else "Audio processing failed",
        }
    except Exception as e:
        logger.error(f"process_audio_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def check_audio_backends_mcp() -> Dict[str, Any]:
    """
    Check which audio generation backends are available in this environment.

    Probes for scipy, soundfile, pedalboard, and pure-Python wave support.
    Returns availability flags and recommended backend.

    Returns:
        Dictionary with backend names and their availability flags.
    """
    try:
        result = check_audio_backends()
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"check_audio_backends_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_audio_generation_options_mcp() -> Dict[str, Any]:
    """
    Return all configurable audio generation options with defaults and ranges.

    Options include: sample rate, duration, frequency range, waveform shape,
    envelope type, stereo/mono, and output format.

    Returns:
        Dictionary with option names, types, defaults, and valid ranges.
    """
    try:
        options = get_audio_generation_options()
        return {"success": True, "options": options}
    except Exception as e:
        logger.error(f"get_audio_generation_options_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def analyze_audio_characteristics_mcp(audio_file_path: str) -> Dict[str, Any]:
    """
    Analyse characteristics of a generated audio file.

    Extracts: duration, sample rate, channel count, peak amplitude,
    RMS energy, zero-crossing rate, and spectral centroid (if scipy is available).

    Args:
        audio_file_path: Absolute or relative path to a WAV/PCM audio file.

    Returns:
        Dictionary with audio characteristics and analysis metadata.
    """
    try:
        result = analyze_audio_characteristics(Path(audio_file_path))
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": True, "characteristics": result}
    except Exception as e:
        logger.error(f"analyze_audio_characteristics_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_audio_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the audio module.

    Includes: module version, available backends, supported output formats,
    GNN sonification capabilities, and list of exported public functions.

    Returns:
        Dictionary with module metadata and feature inventory.
    """
    try:
        info = get_module_info()
        return {"success": True, **info}
    except Exception as e:
        logger.error(f"get_audio_module_info_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def validate_audio_content_mcp(audio_file_path: str) -> Dict[str, Any]:
    """
    Validate an audio file produced by GNN audio processing.

    Checks: file exists, non-zero size, valid WAV header, sample count > 0,
    no clipping (amplitude ≤ 1.0 for float samples).

    Args:
        audio_file_path: Path to the audio file to validate.

    Returns:
        Dictionary with validation result, diagnostics, and any error details.
    """
    try:
        result = validate_audio_content(Path(audio_file_path))
        if isinstance(result, dict):
            return {"success": True, **result}
        return {"success": bool(result), "valid": bool(result)}
    except Exception as e:
        logger.error(f"validate_audio_content_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ──────────────────────────────────────────────────────────


def register_tools(mcp_instance) -> None:
    """Register audio domain tools with the MCP server."""

    mcp_instance.register_tool(
        "process_audio",
        process_audio_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {"type": "string", "description": "Directory containing GNN files"},
                "output_directory": {"type": "string", "description": "Audio output directory"},
                "verbose":          {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run GNN audio processing pipeline: convert GNN models to audio files.",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "check_audio_backends",
        check_audio_backends_mcp,
        {},
        "Check which audio generation backends (scipy, soundfile, pedalboard, wave) are available.",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "get_audio_generation_options",
        get_audio_generation_options_mcp,
        {},
        "Return all configurable audio generation options with defaults and valid ranges.",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "analyze_audio_characteristics",
        analyze_audio_characteristics_mcp,
        {
            "type": "object",
            "properties": {
                "audio_file_path": {"type": "string", "description": "Path to the audio file to analyse"},
            },
            "required": ["audio_file_path"],
        },
        "Analyse characteristics of a GNN-generated audio file (duration, RMS, spectral centroid, etc.).",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "validate_audio_content",
        validate_audio_content_mcp,
        {
            "type": "object",
            "properties": {
                "audio_file_path": {"type": "string", "description": "Path to the audio file to validate"},
            },
            "required": ["audio_file_path"],
        },
        "Validate a GNN-generated audio file: header, sample count, amplitude bounds.",
        module=__package__, category="audio",
    )

    mcp_instance.register_tool(
        "get_audio_module_info",
        get_audio_module_info_mcp,
        {},
        "Return version, feature flags, supported backends and formats of the GNN audio module.",
        module=__package__, category="audio",
    )

    logger.info("audio module MCP tools registered (6 tools).")
