#!/usr/bin/env python3
"""
Audio analyzer module for GNN Processing Pipeline.

This module provides audio analysis functionality.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the audio module and its capabilities."""
    info = {
        'version': "1.0.0",
        'description': "Audio generation and sonification for GNN Processing Pipeline",
        'features': {
            'tonal_generation': True,
            'rhythmic_generation': True,
            'ambient_generation': True,
            'sonification': True,
            'audio_analysis': True,
            'mcp_integration': True
        },
        'audio_capabilities': [],
        'sonification_methods': [],
        'supported_formats': []
    }
    
    # Audio capabilities
    info['audio_capabilities'].extend([
        'Tonal generation',
        'Rhythmic generation',
        'Ambient generation',
        'Sonification',
        'Audio analysis'
    ])
    
    # Sonification methods
    info['sonification_methods'].extend([
        'Variable-to-frequency mapping',
        'Connection-to-rhythm mapping',
        'Model-dynamics-to-ambient mapping'
    ])
    
    # Supported formats
    info['supported_formats'].extend(['wav', 'mp3', 'flac', 'ogg'])
    
    return info

def get_audio_generation_options() -> Dict[str, Any]:
    """Get audio generation options and capabilities.

    Tests expect 'formats', 'effects', and 'backends' keys to exist.
    """
    return {
        'formats': ['wav', 'mp3', 'flac', 'ogg'],
        'effects': ['reverb', 'delay', 'chorus', 'flanger', 'distortion', 'filter'],
        'backends': ['basic', 'sapf', 'pedalboard'],
        # Additional details
        'audio_formats': ['wav', 'mp3', 'flac', 'ogg'],
        'generation_types': ['tonal', 'rhythmic', 'ambient', 'sonification'],
        'sample_rates': [22050, 44100, 48000],
        'bit_depths': [16, 24, 32],
        'channels': [1, 2],
        'duration_options': ['short', 'medium', 'long', 'variable'],
        'oscillators': ['sine', 'square', 'sawtooth', 'triangle', 'noise'],
        'envelopes': ['ADSR', 'AR', 'ASR', 'AD', 'custom'],
        'output_formats': ['wav', 'mp3', 'flac', 'ogg', 'aiff']
    }

def process_gnn_to_audio(gnn_content: str, model_name: str | None = None, output_dir: str | None = None) -> Dict[str, Any]:
    """
    Process GNN content to audio.
    
    Args:
        gnn_content: GNN file content
        model_name: Name of the model
        output_dir: Output directory for audio files
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Validate input
        if not gnn_content.strip():
            return {
                "success": False,
                "error": "Empty GNN content provided"
            }
        
        # Determine output directory if provided
        output_path = Path(output_dir) if output_dir else Path("output/audio")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create processor
        from .classes import SAPFGNNProcessor
        processor = SAPFGNNProcessor()
        
        # Process GNN content
        model_data = processor.process_gnn_content(gnn_content)
        if not model_data["success"]:
            return model_data
        
        # Generate audio
        audio_result = processor.generate_audio(model_data, output_path)
        
        if audio_result["success"]:
            return {
                "success": True,
                "model_name": model_name or "gnn_model",
                "output_dir": str(output_path),
                "audio_files": audio_result["audio_files"]
            }
        else:
            return audio_result
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Audio processing failed: {str(e)}"
        }

def validate_audio_content(audio_content: str) -> bool:
    """Basic validation of audio content payloads expected by tests."""
    return isinstance(audio_content, str) and len(audio_content.strip()) > 0

def convert_gnn_to_sapf(gnn_content: str, output_dir: Path) -> Dict[str, Any]:
    """
    Convert GNN content to SAPF format.
    
    Args:
        gnn_content: GNN file content
        output_dir: Output directory for SAPF files
        
    Returns:
        Dictionary with conversion results
    """
    try:
        # Create processor
        from .processor import SAPFGNNProcessor
        processor = SAPFGNNProcessor()
        
        # Process GNN content
        model_data = processor.process_gnn_content(gnn_content)
        if not model_data["success"]:
            return model_data
        
        # Generate SAPF-specific output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SAPF configuration file
        sapf_config = {
            "model_type": "gnn",
            "audio_engines": processor.audio_engines,
            "supported_formats": processor.supported_formats,
            "variables": model_data.get("variables", []),
            "connections": model_data.get("connections", [])
        }
        
        config_file = output_dir / "sapf_config.json"
        with open(config_file, 'w') as f:
            json.dump(sapf_config, f, indent=2)
        
        return {
            "success": True,
            "output_dir": str(output_dir),
            "sapf_config": str(config_file),
            "model_data": model_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"SAPF conversion failed: {str(e)}"
        }

def generate_audio_from_sapf(sapf_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Generate audio from SAPF configuration.
    
    Args:
        sapf_config: SAPF configuration dictionary
        output_dir: Output directory for audio files
        
    Returns:
        Dictionary with generation results
    """
    try:
        # Create processor
        from .processor import SAPFGNNProcessor
        processor = SAPFGNNProcessor()
        
        # Extract model data from SAPF config
        model_data = {
            "variables": sapf_config.get("variables", []),
            "connections": sapf_config.get("connections", [])
        }
        
        # Generate audio
        audio_result = processor.generate_audio(model_data, output_dir)
        
        if audio_result["success"]:
            return {
                "success": True,
                "output_dir": str(output_dir),
                "audio_files": audio_result["audio_files"],
                "sapf_config": sapf_config
            }
        else:
            return audio_result
            
    except Exception as e:
        return {
            "success": False,
            "error": f"SAPF audio generation failed: {str(e)}"
        }

def validate_sapf_code(sapf_code: str) -> Dict[str, Any]:
    """
    Validate SAPF code.
    
    Args:
        sapf_code: SAPF code string to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Basic validation
        if not sapf_code.strip():
            return {
                "valid": False,
                "errors": ["Empty SAPF code provided"]
            }
        
        # Check for required sections
        required_sections = ["oscillators", "envelopes", "effects"]
        missing_sections = []
        
        for section in required_sections:
            if section not in sapf_code.lower():
                missing_sections.append(section)
        
        if missing_sections:
            return {
                "valid": False,
                "errors": [f"Missing required sections: {', '.join(missing_sections)}"]
            }
        
        return {
            "valid": True,
            "errors": []
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation error: {str(e)}"]
        }
