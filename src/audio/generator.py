"""
Audio Generator Module

Core module for converting GNN models to audio representations using
multiple backends (SAPF, Pedalboard, etc).
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import importlib

logger = logging.getLogger(__name__)

@dataclass
class AudioGenerationResult:
    """Result of audio generation process."""
    success: bool
    audio_file: Optional[Path] = None
    backend_used: str = ""
    duration: float = 0.0
    error_message: str = ""
    visualization_file: Optional[Path] = None
    metadata: Dict[str, Any] = None

def get_available_backends() -> Dict[str, bool]:
    """
    Get information about available audio backends.
    
    Returns:
        Dictionary mapping backend names to availability status
    """
    backends = {
        'sapf': False,
        'pedalboard': False
    }
    
    # Check SAPF availability
    try:
        from .sapf import SAPFGenerator
        backends['sapf'] = True
    except ImportError:
        pass
    
    # Check Pedalboard availability
    try:
        from .pedalboard import PedalboardGenerator
        backends['pedalboard'] = True
    except ImportError:
        pass
    
    return backends

def select_backend(requested_backend: str = 'auto') -> str:
    """
    Select an appropriate audio backend based on availability and request.
    
    Args:
        requested_backend: Requested backend ('auto', 'sapf', 'pedalboard')
        
    Returns:
        Selected backend name
    """
    available = get_available_backends()
    
    if requested_backend == 'auto':
        # Prefer Pedalboard if available, otherwise SAPF
        if available.get('pedalboard', False):
            return 'pedalboard'
        elif available.get('sapf', False):
            return 'sapf'
        else:
            raise ImportError("No audio backends available")
    else:
        # Use specifically requested backend
        if requested_backend not in available:
            raise ValueError(f"Requested backend '{requested_backend}' is not recognized")
        
        if not available.get(requested_backend, False):
            raise ImportError(f"Requested backend '{requested_backend}' is not available")
        
        return requested_backend

def generate_audio(
    gnn_content: str,
    output_file: Union[str, Path],
    model_name: str = "",
    duration: float = 30.0,
    backend: str = 'auto',
    visualization: bool = True,
    **kwargs
) -> bool:
    """
    Generate audio from GNN content using the specified backend.
    
    Args:
        gnn_content: GNN model content
        output_file: Output audio file path
        model_name: Name of the model (extracted from content if not provided)
        duration: Audio duration in seconds
        backend: Backend to use ('auto', 'sapf', 'pedalboard')
        visualization: Whether to create visualizations
        **kwargs: Additional backend-specific parameters
        
    Returns:
        True if generation succeeded, False otherwise
    """
    try:
        # Convert output_file to Path
        if isinstance(output_file, str):
            output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract model_name from content if not provided
        if not model_name:
            import re
            name_match = re.search(r'## ModelName\s*\n([^\n]+)', gnn_content)
            if name_match:
                model_name = name_match.group(1).strip()
            else:
                model_name = output_file.stem
        
        # Select backend
        try:
            selected_backend = select_backend(backend)
        except (ImportError, ValueError) as e:
            logger.error(f"Backend selection failed: {e}")
            return False
        
        logger.info(f"Using audio backend: {selected_backend}")
        
        # Generate audio using selected backend
        if selected_backend == 'sapf':
            from .sapf import SAPFGenerator
            generator = SAPFGenerator()
            result = generator.generate_audio(
                gnn_content=gnn_content,
                output_file=output_file,
                model_name=model_name,
                duration=duration,
                visualization=visualization,
                **kwargs
            )
        elif selected_backend == 'pedalboard':
            from .pedalboard import PedalboardGenerator
            generator = PedalboardGenerator()
            result = generator.generate_audio(
                gnn_content=gnn_content,
                output_file=output_file,
                model_name=model_name,
                duration=duration,
                visualization=visualization,
                **kwargs
            )
        else:
            logger.error(f"Unknown backend: {selected_backend}")
            return False
        
        if not result.success:
            logger.error(f"Audio generation failed with {selected_backend}: {result.error_message}")
            return False
        
        logger.info(f"Successfully generated audio using {selected_backend}: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return False

def get_audio_generation_options() -> Dict[str, Dict[str, str]]:
    """
    Get information about available audio generation options.
    
    Returns:
        Dictionary of available options by category
    """
    options = {
        'backends': {
            'sapf': 'Sound As Pure Form audio generation',
            'pedalboard': 'Spotify Pedalboard audio processing'
        },
        'output_formats': {
            'wav': 'WAV audio format'
        },
        'visualization_types': {
            'waveform': 'Audio waveform visualization',
            'spectrogram': 'Audio spectrogram visualization'
        }
    }
    
    # Add backend-specific options
    available = get_available_backends()
    
    if available.get('sapf', False):
        try:
            from .sapf import SAPFGenerator
            sapf_options = SAPFGenerator.get_options()
            options.update(sapf_options)
        except (ImportError, AttributeError):
            pass
    
    if available.get('pedalboard', False):
        try:
            from .pedalboard import PedalboardGenerator
            pedalboard_options = PedalboardGenerator.get_options()
            options.update(pedalboard_options)
        except (ImportError, AttributeError):
            pass
    
    return options 