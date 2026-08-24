#!/usr/bin/env python3
"""
Audio generator module for GNN Processing Pipeline.

This module provides audio generation functionality.
"""

import math
from typing import Any, Dict, List, Optional, cast

# Optional numpy import with recovery
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = cast(Any, None)
    NUMPY_AVAILABLE = False


def _finite_number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _positive_sample_rate(value: Any, default: int = 44100) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return parsed if parsed > 0 and parsed == value else default


def _audio_array(audio: Any) -> np.ndarray:
    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError("audio samples must be real")
    try:
        samples = np.asarray(audio, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("audio samples must be numeric") from exc
    if samples.ndim not in (1, 2):
        raise ValueError("audio must be mono or frames-by-channels")
    if samples.ndim == 2 and samples.shape[1] < 1:
        raise ValueError("audio must contain at least one channel")
    return np.asarray(np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0))


def generate_tonal_representation(
    variables: List[Dict], connections: List[Dict]
) -> np.ndarray:
    """Generate tonal audio representation of the model."""
    if not variables:
        return np.zeros(int(44100 * 5.0))

    # Create a tonal sequence based on variables
    sample_rate = 44100
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Map variables to frequencies
    base_freq = 440  # A4
    audio = np.zeros_like(t)

    for i, _ in enumerate(variables):
        # Map variable index to frequency
        freq = base_freq * (2 ** (i / 12))  # Chromatic scale
        amplitude = 0.1 / len(variables)  # Normalize amplitude

        # Create tone for this variable
        tone = amplitude * np.sin(2 * np.pi * freq * t)
        audio += tone

    return audio


def generate_rhythmic_representation(
    variables: List[Dict], connections: List[Dict]
) -> np.ndarray:
    """Generate rhythmic audio representation of the model."""
    sample_rate = 44100
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Create rhythmic pattern based on connections
    audio = np.zeros_like(t)

    for i, _ in enumerate(connections):
        # Create rhythmic pulse for each connection
        pulse_freq = 2.0 + (i % 4)  # Different pulse rates
        pulse = np.sin(2 * np.pi * pulse_freq * t) * 0.1

        # Add envelope
        envelope = np.exp(-t * 2)  # Decay envelope
        pulse *= envelope

        audio += pulse

    return audio


def generate_ambient_representation(
    variables: List[Dict], connections: List[Dict]
) -> np.ndarray:
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
    for i, _ in enumerate(variables):
        freq = drone_freq * (i + 2)  # Harmonic series
        harmonic = 0.02 * np.sin(2 * np.pi * freq * t)
        audio += harmonic

    # Add connection-based modulation
    for _ in connections:
        mod_freq = 0.5  # Slow modulation
        modulation = 0.01 * np.sin(2 * np.pi * mod_freq * t)
        audio *= 1 + modulation

    return audio


def generate_sonification_audio(
    dynamics: List[Dict[str, Any]],
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Generate sonification audio from model dynamics, supporting chunked streaming buffering."""
    sample_rate = 44100
    duration = 8.0  # 8 seconds
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, False)

    audio = np.zeros_like(t)

    for i, _ in enumerate(dynamics):
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

    if chunk_size is not None and chunk_size > 0:
        # Buffer into aligned chunk segments (streaming compatibility)
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        pad_len = num_chunks * chunk_size - len(audio)
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))

    return audio


def generate_oscillator_audio(
    frequency: float,
    duration: float,
    oscillator_type: str = "sine",
    sample_rate: int = 44100,
) -> np.ndarray:
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
        config: dict[str, Any] = {
            "frequency": frequency,
            "duration": duration,
            "oscillator_type": oscillator_type,
            "sample_rate": sample_rate,
        }

        return generator.generate_synthetic_audio(config)

    except Exception:
        return np.zeros(int(sample_rate * duration))


def apply_envelope(audio: np.ndarray, envelope_type: str = "ADSR") -> np.ndarray:
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


def mix_audio_channels(channels: List[np.ndarray], mix_mode: str = "add") -> np.ndarray:
    """
    Mix multiple audio channels.

    Args:
        channels: List of audio arrays
        mix_mode: Mixing mode ('add', 'average', 'max')

    Returns:
        Mixed audio array
    """
    if not channels:
        return np.array([], dtype=float)

    arrays = [_audio_array(channel) for channel in channels]
    max_length = max(len(channel) for channel in arrays)
    max_channels = max(
        1 if channel.ndim == 1 else channel.shape[1] for channel in arrays
    )
    multichannel = any(channel.ndim == 2 for channel in arrays)
    padded_channels: List[np.ndarray] = []
    for channel in arrays:
        if max_channels > 1:
            if channel.ndim == 1:
                channel = np.repeat(channel[:, np.newaxis], max_channels, axis=1)
            elif channel.shape[1] == 1:
                channel = np.repeat(channel, max_channels, axis=1)
            elif channel.shape[1] != max_channels:
                raise ValueError("audio arrays must have compatible channel counts")
            padded_multichannel = np.zeros((max_length, max_channels), dtype=float)
            padded_multichannel[: len(channel), :] = channel
            padded_channels.append(padded_multichannel)
        else:
            mono = channel[:, 0] if channel.ndim == 2 else channel
            padded_mono = np.zeros(max_length, dtype=float)
            padded_mono[: len(mono)] = mono
            padded_channels.append(padded_mono)

    if mix_mode == "add":
        mixed = np.sum(padded_channels, axis=0)
    elif mix_mode == "average":
        mixed = np.mean(padded_channels, axis=0)
    elif mix_mode == "max":
        mixed = np.maximum.reduce(padded_channels)
    else:
        mixed = np.sum(padded_channels, axis=0)
    result = np.asarray(mixed)
    if multichannel and result.ndim == 1:
        result = result[:, np.newaxis]
    return result


class SyntheticAudioGenerator:
    """Synthetic Audio Generator for creating artificial sounds."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.supported_formats = ["wav", "mp3", "flac", "ogg"]
        self.oscillator_types = ["sine", "square", "sawtooth", "triangle", "noise"]
        self.envelope_types = ["ADSR", "AR", "ASR", "AD", "custom"]

    def generate_synthetic_audio(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate synthetic audio based on configuration."""
        frequency = _finite_number(config.get("frequency", 440.0), 0.0)
        duration = max(0.0, _finite_number(config.get("duration", 1.0), 0.0))
        sample_rate = _positive_sample_rate(config.get("sample_rate", 44100))
        oscillator_type = str(config.get("oscillator_type", "sine"))

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        if oscillator_type == "sine":
            audio = np.sin(2 * np.pi * frequency * t)
        elif oscillator_type == "square":
            audio = np.sign(np.sin(2 * np.pi * frequency * t))
        elif oscillator_type == "sawtooth":
            audio = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif oscillator_type == "triangle":
            audio = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
        elif oscillator_type == "noise":
            audio = np.random.uniform(-1, 1, len(t))
        else:
            audio = np.sin(2 * np.pi * frequency * t)
        return np.asarray(np.nan_to_num(audio))

    def apply_envelope(
        self, audio: np.ndarray, envelope_type: str = "ADSR"
    ) -> np.ndarray:
        """Apply envelope to audio."""
        samples = _audio_array(audio)
        if envelope_type != "ADSR" or len(samples) == 0:
            return samples
        positions = np.linspace(0.0, 1.0, len(samples))
        envelope = np.interp(
            positions,
            [0.0, 0.1, 0.2, 0.8, 1.0],
            [0.0, 1.0, 0.7, 0.7, 0.0],
        )
        if samples.ndim == 2:
            envelope = envelope[:, np.newaxis]
        return np.asarray(samples * envelope)
