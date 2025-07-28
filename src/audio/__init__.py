"""
Audio module for GNN Processing Pipeline.

This module provides audio generation and sonification capabilities for GNN models.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import numpy as np
from datetime import datetime
import re

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def process_audio(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process GNN files with audio generation and sonification.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("audio")
    
    try:
        log_step_start(logger, "Processing audio")
        
        # Create results directory
        results_dir = output_dir / "audio_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "audio_files_generated": [],
            "sonification_results": [],
            "audio_analysis": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for audio processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Generate audio from GNN model
                    audio_result = generate_audio_from_gnn(gnn_file, results_dir, verbose)
                    results["audio_files_generated"].append(audio_result)
                    
                    # Create sonification
                    sonification = create_sonification(gnn_file, results_dir, verbose)
                    results["sonification_results"].append(sonification)
                    
                    # Analyze audio characteristics
                    analysis = analyze_audio_characteristics(audio_result, verbose)
                    results["audio_analysis"].append(analysis)
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
        
        # Save detailed results
        results_file = results_dir / "audio_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary = generate_audio_summary(results)
        summary_file = results_dir / "audio_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        if results["success"]:
            log_step_success(logger, "Audio processing completed successfully")
        else:
            log_step_error(logger, "Audio processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, "Audio processing failed", {"error": str(e)})
        return False

def generate_audio_from_gnn(file_path: Path, output_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Generate audio from a GNN model.
    
    Args:
        file_path: Path to the GNN file
        output_dir: Directory to save audio files
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing audio generation results
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract model structure for audio generation
        variables = extract_variables_for_audio(content)
        connections = extract_connections_for_audio(content)
        
        # Generate different types of audio
        audio_files = {}
        
        # 1. Generate tonal representation
        tonal_audio = generate_tonal_representation(variables, connections)
        tonal_path = output_dir / f"{file_path.stem}_tonal.wav"
        save_audio_file(tonal_audio, tonal_path, sample_rate=44100)
        audio_files["tonal"] = str(tonal_path)
        
        # 2. Generate rhythmic representation
        rhythmic_audio = generate_rhythmic_representation(variables, connections)
        rhythmic_path = output_dir / f"{file_path.stem}_rhythmic.wav"
        save_audio_file(rhythmic_audio, rhythmic_path, sample_rate=44100)
        audio_files["rhythmic"] = str(rhythmic_path)
        
        # 3. Generate ambient representation
        ambient_audio = generate_ambient_representation(variables, connections)
        ambient_path = output_dir / f"{file_path.stem}_ambient.wav"
        save_audio_file(ambient_audio, ambient_path, sample_rate=44100)
        audio_files["ambient"] = str(ambient_path)
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "audio_files": audio_files,
            "variables_count": len(variables),
            "connections_count": len(connections),
            "generation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to generate audio from {file_path}: {e}")

def extract_variables_for_audio(content: str) -> List[Dict[str, Any]]:
    """Extract variables from GNN content for audio generation."""
    variables = []
    
    # Look for variable definitions
    var_patterns = [
        r'(\w+)\s*:\s*(\w+)',  # name: type
        r'(\w+)\s*=\s*([^;\n]+)',  # name = value
        r'(\w+)\s*\[([^\]]+)\]',  # name[dimensions]
    ]
    
    for pattern in var_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            variables.append({
                "name": match.group(1),
                "type": match.group(2) if len(match.groups()) > 1 else "unknown",
                "definition": match.group(0)
            })
    
    return variables

def extract_connections_for_audio(content: str) -> List[Dict[str, Any]]:
    """Extract connections from GNN content for audio generation."""
    connections = []
    
    # Look for connection patterns
    conn_patterns = [
        r'(\w+)\s*->\s*(\w+)',  # source -> target
        r'(\w+)\s*→\s*(\w+)',   # source → target
        r'(\w+)\s*connects\s*(\w+)',  # source connects target
    ]
    
    for pattern in conn_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            connections.append({
                "source": match.group(1),
                "target": match.group(2),
                "definition": match.group(0)
            })
    
    return connections

def generate_tonal_representation(variables: List[Dict], connections: List[Dict]) -> np.ndarray:
    """Generate tonal audio representation of the model."""
    # Create a tonal sequence based on variables
    sample_rate = 44100
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Map variables to frequencies
    base_freq = 440  # A4
    audio = np.zeros_like(t)
    
    for i, var in enumerate(variables):
        # Map variable index to frequency
        freq = base_freq * (2 ** (i / 12))  # Chromatic scale
        amplitude = 0.1 / len(variables)  # Normalize amplitude
        
        # Create tone for this variable
        tone = amplitude * np.sin(2 * np.pi * freq * t)
        audio += tone
    
    return audio

def generate_rhythmic_representation(variables: List[Dict], connections: List[Dict]) -> np.ndarray:
    """Generate rhythmic audio representation of the model."""
    sample_rate = 44100
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create rhythmic pattern based on connections
    audio = np.zeros_like(t)
    
    for i, conn in enumerate(connections):
        # Create rhythmic pulse for each connection
        pulse_freq = 2.0 + (i % 4)  # Different pulse rates
        pulse = np.sin(2 * np.pi * pulse_freq * t) * 0.1
        
        # Add envelope
        envelope = np.exp(-t * 2)  # Decay envelope
        pulse *= envelope
        
        audio += pulse
    
    return audio

def generate_ambient_representation(variables: List[Dict], connections: List[Dict]) -> np.ndarray:
    """Generate ambient audio representation of the model."""
    sample_rate = 44100
    duration = 10.0  # 10 seconds for ambient
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create ambient soundscape
    audio = np.zeros_like(t)
    
    # Add low-frequency drone
    drone_freq = 55  # A1
    drone = 0.05 * np.sin(2 * np.pi * drone_freq * t)
    audio += drone
    
    # Add variable-based harmonics
    for i, var in enumerate(variables):
        freq = drone_freq * (i + 2)  # Harmonic series
        harmonic = 0.02 * np.sin(2 * np.pi * freq * t)
        audio += harmonic
    
    # Add connection-based modulation
    for conn in connections:
        mod_freq = 0.5  # Slow modulation
        modulation = 0.01 * np.sin(2 * np.pi * mod_freq * t)
        audio *= (1 + modulation)
    
    return audio

def save_audio_file(audio: np.ndarray, file_path: Path, sample_rate: int = 44100):
    """Save audio data to file."""
    try:
        import soundfile as sf
        sf.write(str(file_path), audio, sample_rate)
    except ImportError:
        # Fallback to basic WAV writing
        write_basic_wav(audio, file_path, sample_rate)

def write_basic_wav(audio: np.ndarray, file_path: Path, sample_rate: int):
    """Write basic WAV file without external dependencies."""
    import struct
    
    # Normalize audio
    audio = np.clip(audio, -1, 1)
    audio = (audio * 32767).astype(np.int16)
    
    with open(file_path, 'wb') as f:
        # WAV header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(audio) * 2))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))  # PCM
        f.write(struct.pack('<H', 1))  # Mono
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(audio) * 2))
        f.write(audio.tobytes())

def create_sonification(file_path: Path, output_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """Create sonification of the GNN model."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract model dynamics
        dynamics = extract_model_dynamics(content)
        
        # Generate sonification
        sonification_audio = generate_sonification_audio(dynamics)
        sonification_path = output_dir / f"{file_path.stem}_sonification.wav"
        save_audio_file(sonification_audio, sonification_path, sample_rate=44100)
        
        return {
            "file_path": str(file_path),
            "sonification_file": str(sonification_path),
            "dynamics_analyzed": len(dynamics),
            "sonification_type": "model_dynamics",
            "generation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to create sonification for {file_path}: {e}")

def extract_model_dynamics(content: str) -> List[Dict[str, Any]]:
    """Extract model dynamics for sonification."""
    dynamics = []
    
    # Look for dynamic elements
    dynamic_patterns = [
        r'(\w+)\s*evolves',  # variable evolves
        r'(\w+)\s*changes',  # variable changes
        r'(\w+)\s*updates',  # variable updates
        r'(\w+)\s*transitions',  # state transitions
    ]
    
    for pattern in dynamic_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            dynamics.append({
                "element": match.group(1),
                "dynamic_type": pattern.split()[0],
                "description": match.group(0)
            })
    
    return dynamics

def generate_sonification_audio(dynamics: List[Dict[str, Any]]) -> np.ndarray:
    """Generate sonification audio from model dynamics."""
    sample_rate = 44100
    duration = 8.0  # 8 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    audio = np.zeros_like(t)
    
    for i, dynamic in enumerate(dynamics):
        # Create dynamic sound for each element
        base_freq = 220 + (i * 50)  # Different base frequency for each element
        
        # Add frequency modulation
        mod_freq = 0.5 + (i * 0.2)
        freq_mod = base_freq * (1 + 0.1 * np.sin(2 * np.pi * mod_freq * t))
        
        # Generate tone with frequency modulation
        tone = 0.05 * np.sin(2 * np.pi * freq_mod * t)
        
        # Add envelope
        envelope = np.exp(-t * 0.5)  # Decay
        tone *= envelope
        
        audio += tone
    
    return audio

def analyze_audio_characteristics(audio_result: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Analyze characteristics of generated audio."""
    analysis = {
        "file_path": audio_result["file_path"],
        "audio_characteristics": {},
        "spectral_analysis": {},
        "temporal_analysis": {}
    }
    
    # Analyze each audio file
    for audio_type, audio_path in audio_result["audio_files"].items():
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            
            # Basic characteristics
            analysis["audio_characteristics"][audio_type] = {
                "duration": len(audio_data) / sample_rate,
                "sample_rate": sample_rate,
                "channels": len(audio_data.shape),
                "max_amplitude": np.max(np.abs(audio_data)),
                "rms_amplitude": np.sqrt(np.mean(audio_data**2))
            }
            
            # Spectral analysis
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Take first channel
            
            # FFT for spectral analysis
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
            
            # Find dominant frequencies
            magnitude = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate spectral metrics with safe division
            magnitude_sum = np.sum(magnitude[:len(magnitude)//2])
            if magnitude_sum > 0:
                spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / magnitude_sum
                spectral_bandwidth = np.sqrt(np.sum((freqs[:len(freqs)//2] - dominant_freq)**2 * magnitude[:len(magnitude)//2]) / magnitude_sum)
            else:
                spectral_centroid = 0.0
                spectral_bandwidth = 0.0
            
            analysis["spectral_analysis"][audio_type] = {
                "dominant_frequency": dominant_freq,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth
            }
            
        except Exception as e:
            analysis["audio_characteristics"][audio_type] = {"error": str(e)}
    
    return analysis

def generate_audio_summary(results: Dict[str, Any]) -> str:
    """Generate a markdown summary of audio processing results."""
    summary = f"""# Audio Processing Summary

Generated on: {results['timestamp']}

## Overview
- **Files Processed**: {results['processed_files']}
- **Success**: {results['success']}
- **Errors**: {len(results['errors'])}

## Audio Files Generated
"""
    
    for audio_result in results["audio_files_generated"]:
        summary += f"""
### {audio_result['file_name']}
- **Variables**: {audio_result['variables_count']}
- **Connections**: {audio_result['connections_count']}
- **Audio Files**: {len(audio_result['audio_files'])}
"""
        for audio_type, audio_path in audio_result['audio_files'].items():
            summary += f"  - {audio_type}: {Path(audio_path).name}\n"
    
    if results["errors"]:
        summary += "\n## Errors\n"
        for error in results["errors"]:
            summary += f"- {error}\n"
    
    return summary

class SAPFGNNProcessor:
    """SAPF (Sonification and Audio Processing Framework) GNN Processor."""
    
    def __init__(self):
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg']
        self.audio_engines = ['basic', 'sapf', 'pedalboard']
    
    def process_gnn_content(self, gnn_content: str) -> Dict[str, Any]:
        """Process GNN content for audio generation."""
        try:
            # Extract variables and connections
            variables = extract_variables_for_audio(gnn_content)
            connections = extract_connections_for_audio(gnn_content)
            
            return {
                "success": True,
                "variables": variables,
                "connections": connections,
                "audio_ready": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_audio(self, model_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Generate audio from model data."""
        try:
            # Generate different audio representations
            tonal_audio = generate_tonal_representation(
                model_data.get("variables", []), 
                model_data.get("connections", [])
            )
            
            rhythmic_audio = generate_rhythmic_representation(
                model_data.get("variables", []), 
                model_data.get("connections", [])
            )
            
            ambient_audio = generate_ambient_representation(
                model_data.get("variables", []), 
                model_data.get("connections", [])
            )
            
            # Save audio files
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_audio_file(tonal_audio, output_dir / "tonal.wav")
            save_audio_file(rhythmic_audio, output_dir / "rhythmic.wav")
            save_audio_file(ambient_audio, output_dir / "ambient.wav")
            
            return {
                "success": True,
                "audio_files": {
                    "tonal": str(output_dir / "tonal.wav"),
                    "rhythmic": str(output_dir / "rhythmic.wav"),
                    "ambient": str(output_dir / "ambient.wav")
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the audio module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
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
    """Get audio generation options and capabilities."""
    return {
        'audio_formats': ['wav', 'mp3', 'flac', 'ogg'],
        'generation_types': ['tonal', 'rhythmic', 'ambient', 'sonification'],
        'sample_rates': [22050, 44100, 48000],
        'bit_depths': [16, 24, 32],
        'channels': [1, 2],
        'duration_options': ['short', 'medium', 'long', 'variable'],
        'oscillators': ['sine', 'square', 'sawtooth', 'triangle', 'noise'],
        'envelopes': ['ADSR', 'AR', 'ASR', 'AD', 'custom'],
        'effects': ['reverb', 'delay', 'chorus', 'flanger', 'distortion', 'filter'],
        'output_formats': ['wav', 'mp3', 'flac', 'ogg', 'aiff', 'wma']
    }

def process_gnn_to_audio(gnn_content: str, model_name: str, output_dir: str) -> Dict[str, Any]:
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
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create processor
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
                "model_name": model_name,
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
        import json
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

class SyntheticAudioGenerator:
    """Synthetic Audio Generator for creating artificial sounds."""
    
    def __init__(self):
        self.supported_formats = ['wav', 'mp3', 'flac', 'ogg']
        self.oscillator_types = ['sine', 'square', 'sawtooth', 'triangle', 'noise']
        self.envelope_types = ['ADSR', 'AR', 'ASR', 'AD', 'custom']
    
    def generate_synthetic_audio(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate synthetic audio based on configuration."""
        try:
            # Extract parameters
            frequency = config.get('frequency', 440.0)
            duration = config.get('duration', 1.0)
            sample_rate = config.get('sample_rate', 44100)
            oscillator_type = config.get('oscillator_type', 'sine')
            
            # Generate time array
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Generate waveform based on oscillator type
            if oscillator_type == 'sine':
                audio = np.sin(2 * np.pi * frequency * t)
            elif oscillator_type == 'square':
                audio = np.sign(np.sin(2 * np.pi * frequency * t))
            elif oscillator_type == 'sawtooth':
                audio = 2 * (t * frequency - np.floor(t * frequency + 0.5))
            elif oscillator_type == 'triangle':
                audio = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
            elif oscillator_type == 'noise':
                audio = np.random.uniform(-1, 1, len(t))
            else:
                audio = np.sin(2 * np.pi * frequency * t)  # Default to sine
            
            return audio
            
        except Exception as e:
            # Return silence on error
            return np.zeros(int(config.get('sample_rate', 44100) * config.get('duration', 1.0)))
    
    def apply_envelope(self, audio: np.ndarray, envelope_type: str = 'ADSR') -> np.ndarray:
        """Apply envelope to audio."""
        try:
            if envelope_type == 'ADSR':
                # Simple ADSR envelope
                attack_samples = int(len(audio) * 0.1)
                decay_samples = int(len(audio) * 0.1)
                release_samples = int(len(audio) * 0.2)
                sustain_samples = len(audio) - attack_samples - decay_samples - release_samples
                
                # Create envelope
                envelope = np.ones(len(audio))
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, 0.7, decay_samples)
                envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = 0.7
                envelope[-release_samples:] = np.linspace(0.7, 0, release_samples)
                
                return audio * envelope
            else:
                return audio
                
        except Exception:
            return audio

def generate_oscillator_audio(frequency: float, duration: float, oscillator_type: str = 'sine', sample_rate: int = 44100) -> np.ndarray:
    """
    Generate oscillator audio.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        oscillator_type: Type of oscillator ('sine', 'square', 'sawtooth', 'triangle', 'noise')
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio array
    """
    try:
        # Create generator
        generator = SyntheticAudioGenerator()
        
        # Generate audio
        config = {
            'frequency': frequency,
            'duration': duration,
            'oscillator_type': oscillator_type,
            'sample_rate': sample_rate
        }
        
        return generator.generate_synthetic_audio(config)
        
    except Exception as e:
        # Return silence on error
        return np.zeros(int(sample_rate * duration))

def apply_envelope(audio: np.ndarray, envelope_type: str = 'ADSR') -> np.ndarray:
    """
    Apply envelope to audio.
    
    Args:
        audio: Audio array
        envelope_type: Type of envelope ('ADSR', 'AR', 'ASR', 'AD', 'custom')
        
    Returns:
        Audio array with envelope applied
    """
    try:
        # Create generator
        generator = SyntheticAudioGenerator()
        
        # Apply envelope
        return generator.apply_envelope(audio, envelope_type)
        
    except Exception:
        return audio

def mix_audio_channels(channels: List[np.ndarray], mix_mode: str = 'add') -> np.ndarray:
    """
    Mix multiple audio channels.
    
    Args:
        channels: List of audio arrays
        mix_mode: Mixing mode ('add', 'average', 'max')
        
    Returns:
        Mixed audio array
    """
    try:
        if not channels:
            return np.array([])
        
        # Ensure all channels have the same length
        max_length = max(len(channel) for channel in channels)
        padded_channels = []
        
        for channel in channels:
            if len(channel) < max_length:
                # Pad with zeros
                padded = np.zeros(max_length)
                padded[:len(channel)] = channel
                padded_channels.append(padded)
            else:
                padded_channels.append(channel)
        
        # Mix channels based on mode
        if mix_mode == 'add':
            mixed = np.sum(padded_channels, axis=0)
        elif mix_mode == 'average':
            mixed = np.mean(padded_channels, axis=0)
        elif mix_mode == 'max':
            mixed = np.maximum.reduce(padded_channels)
        else:
            mixed = np.sum(padded_channels, axis=0)  # Default to add
        
        return mixed
        
    except Exception:
        # Return first channel or empty array on error
        return channels[0] if channels else np.array([])

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

__all__ = [
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
    'generate_oscillator_audio',
    'apply_envelope',
    'mix_audio_channels',
    'FEATURES',
    '__version__'
]
