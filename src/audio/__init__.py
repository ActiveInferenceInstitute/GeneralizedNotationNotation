"""
Audio Generation Module

This module provides tools for converting GNN models to audio representations,
enabling auditory exploration and debugging of Active Inference generative models
using multiple audio backends.
"""

from .generator import (
    generate_audio,
    get_available_backends,
    AudioGenerationResult
)

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Audio generation from GNN models"

# Feature availability flags
FEATURES = {
    'sapf_backend': True,
    'pedalboard_backend': False,  # Will be set to True if available
    'audio_generation': True,
    'audio_visualization': True
}

# Try to import backends to determine availability
try:
    from .pedalboard import PedalboardGenerator
    FEATURES['pedalboard_backend'] = True
except ImportError:
    pass

# Main API functions
__all__ = [
    # Core audio processing
    "generate_audio",
    "get_available_backends",
    "AudioGenerationResult",
    
    # Metadata
    "FEATURES",
    "__version__"
]

def get_module_info():
    """Get comprehensive information about the audio module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'audio_capabilities': [],
        'supported_backends': get_available_backends(),
        'supported_formats': []
    }
    
    # Audio capabilities
    info['audio_capabilities'].extend([
        'GNN to audio conversion',
        'Audio file generation',
        'Synthetic audio synthesis',
        'Audio visualization',
        'Multi-backend support'
    ])
    
    # Supported formats
    info['supported_formats'].extend(['WAV', 'GNN'])
    
    return info 