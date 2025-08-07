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

# Import processor functions
from .processor import (
    process_gnn_to_audio,
    generate_sapf_audio,
    create_sapf_visualization,
    generate_sapf_report
)

# Import utility functions
from .utils import (
    get_module_info,
    get_audio_generation_options,
    register_tools
)

# MCP integration
try:
    from ..mcp import (
        register_sapf_tools,
        handle_convert_gnn_to_sapf_audio,
        handle_generate_sapf_code,
        handle_validate_sapf_syntax,
        handle_generate_audio_from_sapf,
        handle_analyze_gnn_for_audio
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Module metadata
__version__ = "1.1.0"
__author__ = "Active Inference Institute"
__description__ = "SAPF audio generation from GNN models"

# Feature availability flags
FEATURES = {
    'gnn_to_sapf_conversion': True,
    'audio_generation': True,
    'sapf_validation': True,
    'synthetic_audio': True,
    'mcp_integration': MCP_AVAILABLE
}

# Main API functions
__all__ = [
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
    
    # MCP integration (if available)
    "register_sapf_tools",
    "handle_convert_gnn_to_sapf_audio",
    "handle_generate_sapf_code",
    "handle_validate_sapf_syntax",
    "handle_generate_audio_from_sapf",
    "handle_analyze_gnn_for_audio",
    
    # Metadata
    "FEATURES",
    "__version__"
]

# Add conditional exports
if not MCP_AVAILABLE:
    __all__.remove("register_sapf_tools")
    __all__.remove("handle_convert_gnn_to_sapf_audio")
    __all__.remove("handle_generate_sapf_code")
    __all__.remove("handle_validate_sapf_syntax")
    __all__.remove("handle_generate_audio_from_sapf")
    __all__.remove("handle_analyze_gnn_for_audio") 