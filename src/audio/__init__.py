"""
Audio module for GNN Processing Pipeline.

This module provides audio generation and sonification capabilities for GNN models.
"""

import logging
from typing import Any, Dict

# Configure logger
logger = logging.getLogger(__name__)

# Essential imports
from .analyzer import (
    convert_gnn_to_sapf,
    generate_audio_from_sapf,
    get_audio_generation_options,
    get_module_info,
    process_gnn_to_audio,
    validate_audio_content,
    validate_sapf_code,
)
from .classes import AudioGenerator, SAPFGNNProcessor
from .generator import (
    SyntheticAudioGenerator,
    apply_envelope,
    generate_ambient_representation,
    generate_oscillator_audio,
    generate_rhythmic_representation,
    generate_sonification_audio,
    generate_tonal_representation,
    mix_audio_channels,
)
from .processor import (
    analyze_audio_characteristics,
    create_sonification,
    extract_connections_for_audio,
    extract_model_dynamics,
    extract_variables_for_audio,
    generate_audio_from_gnn,
    generate_audio_summary,
    process_audio,
    save_audio_file,
    write_basic_wav,
)

# Backwards-compatible aliases expected by tests
SAPFProcessor = SAPFGNNProcessor
PedalboardProcessor = SAPFGNNProcessor

# Module metadata
__version__ = "1.1.3"
__author__ = "Active Inference Institute"
__description__ = "Audio generation and sonification for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'tonal_generation': True,
    'rhythmic_generation': True,
    'ambient_generation': True,
    'sonification': True,
    'audio_analysis': True,
    'mcp_integration': True
}

def check_audio_backends() -> Dict[str, Any]:
    """Check availability of audio backends."""
    backends = {}

    # Check librosa
    try:
        import librosa
        backends['librosa'] = {
            'available': True,
            'version': librosa.__version__
        }
    except ImportError:
        backends['librosa'] = {'available': False, 'version': None}

    # Check soundfile
    try:
        import soundfile
        backends['soundfile'] = {
            'available': True,
            'version': soundfile.__version__
        }
    except ImportError:
        backends['soundfile'] = {'available': False, 'version': None}

    # Check pedalboard
    try:
        import pedalboard
        backends['pedalboard'] = {
            'available': True,
            'version': pedalboard.__version__
        }
    except ImportError:
        backends['pedalboard'] = {'available': False, 'version': None}

    # Check numpy (always needed for audio generation)
    try:
        import numpy
        backends['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        backends['numpy'] = {'available': False, 'version': None}

    return backends

__all__ = [
    'AudioGenerator',
    'process_audio',
    'generate_audio_from_gnn',
    'create_sonification',
    'analyze_audio_characteristics',
    'SAPFGNNProcessor',
    'SyntheticAudioGenerator',
    'get_module_info',
    'get_audio_generation_options',
    'process_gnn_to_audio',
    'convert_gnn_to_sapf',
    'generate_audio_from_sapf',
    'validate_sapf_code',
    'validate_audio_content',
    'generate_oscillator_audio',
    'apply_envelope',
    'mix_audio_channels',
    'check_audio_backends',
    'FEATURES',
    '__version__'
]
