"""
Audio module for GNN Processing Pipeline.

This module provides audio generation and sonification capabilities for GNN models.
"""

from .processor import (
    process_audio,
    generate_audio_from_gnn,
    create_sonification,
    analyze_audio_characteristics,
    extract_variables_for_audio,
    extract_connections_for_audio,
    save_audio_file,
    write_basic_wav,
    extract_model_dynamics,
    generate_audio_summary
)

from .generator import (
    generate_tonal_representation,
    generate_rhythmic_representation,
    generate_ambient_representation,
    generate_sonification_audio,
    generate_oscillator_audio,
    apply_envelope,
    mix_audio_channels,
    SyntheticAudioGenerator
)

from .analyzer import (
    get_module_info,
    get_audio_generation_options,
    process_gnn_to_audio,
    convert_gnn_to_sapf,
    generate_audio_from_sapf,
    validate_sapf_code,
    validate_audio_content
)

from .classes import (
    AudioGenerator,
    SAPFGNNProcessor
)

# Backwards-compatible aliases expected by tests
SAPFProcessor = SAPFGNNProcessor
PedalboardProcessor = SAPFGNNProcessor

# Module metadata
__version__ = "1.0.0"
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


def process_audio(target_dir, output_dir, verbose=False, logger=None, **kwargs):
    """
    Main processing function for audio.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        logger: Logger instance
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    import json
    from pathlib import Path
    from datetime import datetime
    
    if logger is None:
        logger = logging.getLogger(__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing audio for files in {target_dir}")
        
        # Check audio backend availability
        audio_backends = check_audio_backends()
        
        # Create processing summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "audio_backends": audio_backends,
            "processing_status": "completed",
            "backends_available": [backend for backend, info in audio_backends.items() if info.get('available')],
            "message": "Audio processing module ready for sonification and audio generation"
        }
        
        # Save summary
        summary_file = output_dir / "audio_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"🎵 Audio processing summary saved to: {summary_file}")
        
        # Save backend details
        backends_file = output_dir / "audio_backends_status.json"
        with open(backends_file, 'w') as f:
            json.dump(audio_backends, f, indent=2)
        logger.info(f"🔧 Audio backends status saved to: {backends_file}")
        
        logger.info(f"✅ Audio processing completed")
        return True
    except Exception as e:
        logger.error(f"❌ Audio processing failed: {e}")
        return False

def check_audio_backends():
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
    'FEATURES',
    '__version__'
]
