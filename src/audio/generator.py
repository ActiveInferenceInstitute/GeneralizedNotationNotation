#!/usr/bin/env python3
"""
Audio generator module for GNN Processing Pipeline.

This module provides audio generation functionality.
"""

import numpy as np
from typing import Dict, Any, List
from pathlib import Path

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