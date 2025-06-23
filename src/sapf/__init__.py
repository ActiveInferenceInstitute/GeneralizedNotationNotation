"""
SAPF (Sound As Pure Form) Integration Module

This module provides tools for converting GNN models to SAPF audio representations,
enabling auditory exploration and debugging of Active Inference generative models.
"""

from .sapf_gnn_processor import (
    SAPFGNNProcessor,
    convert_gnn_to_sapf,
    generate_audio_from_sapf,
    validate_sapf_code
)

from .audio_generators import (
    SyntheticAudioGenerator,
    generate_oscillator_audio,
    apply_envelope,
    mix_audio_channels
)

__version__ = "1.0.0"
__all__ = [
    "SAPFGNNProcessor",
    "convert_gnn_to_sapf", 
    "generate_audio_from_sapf",
    "validate_sapf_code",
    "SyntheticAudioGenerator",
    "generate_oscillator_audio",
    "apply_envelope",
    "mix_audio_channels"
] 