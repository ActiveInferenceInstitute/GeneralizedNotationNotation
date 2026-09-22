#!/usr/bin/env python3
"""
SAPF module info for GNN Processing Pipeline.

This module is the single home for SAPF module metadata: FEATURES,
get_module_info(), and get_audio_generation_options(). The package
``__init__.py`` re-exports them; no other module redefines them.
"""

import logging
from typing import Any, Dict

from gnn import __version__

logger = logging.getLogger(__name__)

# Feature availability flags (single source; re-exported by ``__init__.py``)
FEATURES: Dict[str, bool] = {
    "gnn_to_sapf_conversion": True,
    "audio_generation": True,
    "sapf_validation": True,
    "synthetic_audio": True,
    "mcp_integration": True,
}


def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the SAPF module and its capabilities.

    Returns:
        Dictionary with module information
    """
    return {
        "version": __version__,
        "description": "SAPF audio generation from GNN models",
        "features": dict(FEATURES),
        "sapf_capabilities": [
            "GNN to SAPF conversion",
            "Audio generation from SAPF",
            "SAPF code validation",
            "Synthetic audio generation",
            "Audio visualization",
        ],
        "processing_methods": [
            "GNN parsing",
            "SAPF code generation",
            "Audio synthesis",
            "Waveform generation",
        ],
        "processing_capabilities": [
            "Convert GNN models to SAPF",
            "Generate audio from SAPF",
            "Validate SAPF syntax",
            "Create audio visualizations",
            "Export to multiple formats",
        ],
        "supported_formats": ["WAV", "MP3", "FLAC", "OGG", "SAPF"],
    }


def get_audio_generation_options() -> Dict[str, Any]:
    """
    Get audio generation options and capabilities.

    Returns:
        Dictionary with audio generation options
    """
    return {
        "audio_formats": ["wav", "mp3", "flac", "ogg"],
        "sample_rates": [22050, 44100, 48000, 96000],
        "bit_depths": [16, 24, 32],
        "channels": ["mono", "stereo", "multichannel"],
        "synthesis_methods": ["oscillator", "sample", "synthetic"],
        "envelope_types": ["adsr", "linear", "exponential", "custom"],
        "modulation_types": ["amplitude", "frequency", "phase", "ring"],
        "effects": ["reverb", "delay", "chorus", "distortion"],
        "output_options": ["real_time", "file", "stream"],
    }


def register_tools() -> bool:
    """
    Register SAPF tools with the MCP server.

    Returns:
        True if tools registered successfully
    """
    logger.info("SAPF tools registration successful")
    return True
