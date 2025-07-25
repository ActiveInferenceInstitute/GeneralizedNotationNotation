"""
SAPF (Sound As Pure Form) Module

This module provides a redirect to the actual SAPF implementation in audio.sapf
for backward compatibility with tests and external imports.
"""

# Import all required functions from the actual audio.sapf module
from audio.sapf import (
    # Core SAPF processing functions that tests expect
    convert_gnn_to_sapf,
    generate_sapf_audio,
    validate_sapf_code,
    create_sapf_visualization,
    generate_sapf_report,
    
    # Additional functions for completeness
    generate_audio_from_sapf,
    SAPFGNNProcessor,
    SyntheticAudioGenerator,
    generate_oscillator_audio,
    apply_envelope,
    mix_audio_channels,
    
    # Module metadata
    FEATURES,
    __version__ as sapf_version,
    
    # Utility functions
    get_module_info,
    process_gnn_to_audio,
    get_audio_generation_options
)

# Re-export everything for backward compatibility
__all__ = [
    # Core functions expected by tests
    "convert_gnn_to_sapf",
    "generate_sapf_audio", 
    "validate_sapf_code",
    "create_sapf_visualization",
    "generate_sapf_report",
    
    # Additional SAPF functionality
    "generate_audio_from_sapf",
    "SAPFGNNProcessor",
    "SyntheticAudioGenerator",
    "generate_oscillator_audio",
    "apply_envelope",
    "mix_audio_channels",
    
    # Metadata and utilities
    "FEATURES",
    "sapf_version",
    "get_module_info",
    "process_gnn_to_audio",
    "get_audio_generation_options"
]

# Module metadata
__version__ = sapf_version
__author__ = "Active Inference Institute"
__description__ = "SAPF audio generation from GNN models (compatibility layer)" 