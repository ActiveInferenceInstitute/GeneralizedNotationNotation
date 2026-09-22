"""
SAPF (Sound As Pure Form) Integration Module

This module provides tools for converting GNN models to SAPF audio representations,
enabling auditory exploration and debugging of Active Inference generative models.
"""

from typing import Any

# Package version: re-exported from the gnn package (single version source).
from gnn import __version__

from .audio_generators import (
    SyntheticAudioGenerator,
    apply_envelope,
    generate_oscillator_audio,
    mix_audio_channels,
)

# Import module introspection helpers + single-source module metadata
from .module_info import (
    FEATURES,
    get_audio_generation_options,
    get_module_info,
    register_tools,
)

# Import processor functions
from .processor import (
    create_sapf_visualization,
    generate_sapf_audio,
    generate_sapf_report,
    process_gnn_to_audio,
)
from .sapf_gnn_processor import (
    SAPFGNNProcessor,
    convert_gnn_to_sapf,
    generate_audio_from_sapf,
    validate_sapf_code,
)

__author__ = "Active Inference Institute"
__description__ = "SAPF audio generation from GNN models"

# Main API functions

__all__: list[Any] = [
    # Core SAPF processing
    "SAPFGNNProcessor",
    "convert_gnn_to_sapf",
    "generate_audio_from_sapf",
    "validate_sapf_code",
    # Processor functions
    "process_gnn_to_audio",
    "generate_sapf_audio",
    "create_sapf_visualization",
    "generate_sapf_report",
    # Audio generation
    "SyntheticAudioGenerator",
    "generate_oscillator_audio",
    "apply_envelope",
    "mix_audio_channels",
    # Utility functions
    "get_module_info",
    "get_audio_generation_options",
    "register_tools",
    # Metadata
    "FEATURES",
    "__version__",
]
