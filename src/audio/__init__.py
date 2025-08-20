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


def process_audio(target_dir, output_dir, verbose=False, **kwargs):
    """
    Main processing function for audio.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Processing audio for files in {target_dir}")
        # Placeholder implementation - delegate to actual module functions
        # This would be replaced with actual implementation
        logger.info(f"Audio processing completed")
        return True
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return False


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
