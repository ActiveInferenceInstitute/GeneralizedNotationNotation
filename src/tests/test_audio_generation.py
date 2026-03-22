#!/usr/bin/env python3
"""
Tests for audio/generator.py — real function-level coverage.

Tests generate_tonal_representation, generate_rhythmic_representation,
generate_ambient_representation, generate_sonification_audio,
generate_oscillator_audio, mix_audio_channels, and SyntheticAudioGenerator.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy required")


@pytest.fixture(autouse=True)
def require_audio():
    try:
        import audio.generator  # noqa: F401
    except ImportError:
        pytest.skip("audio.generator not available")


class TestGenerateTonalRepresentation:
    def test_empty_variables_returns_silence(self):
        from audio.generator import generate_tonal_representation
        result = generate_tonal_representation([], [])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == int(44100 * 5.0)
        assert np.all(result == 0.0)

    def test_single_variable_produces_nonzero_audio(self):
        from audio.generator import generate_tonal_representation
        variables = [{"name": "s"}]
        result = generate_tonal_representation(variables, [])
        assert result.shape[0] > 0
        assert np.any(result != 0.0)

    def test_multiple_variables_returns_correct_length(self):
        from audio.generator import generate_tonal_representation
        variables = [{"name": f"v{i}"} for i in range(4)]
        result = generate_tonal_representation(variables, [])
        assert result.shape[0] == int(44100 * 5.0)

    def test_connections_do_not_change_output_length(self):
        from audio.generator import generate_tonal_representation
        variables = [{"name": "s"}, {"name": "o"}]
        connections = [{"source": "s", "target": "o"}]
        result = generate_tonal_representation(variables, connections)
        assert result.shape[0] == int(44100 * 5.0)


class TestGenerateRhythmicRepresentation:
    def test_no_connections_returns_silence(self):
        from audio.generator import generate_rhythmic_representation
        result = generate_rhythmic_representation([], [])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == int(44100 * 5.0)
        assert np.all(result == 0.0)

    def test_single_connection_produces_audio(self):
        from audio.generator import generate_rhythmic_representation
        result = generate_rhythmic_representation([], [{"source": "a", "target": "b"}])
        assert result.shape[0] > 0
        assert np.any(result != 0.0)

    def test_multiple_connections_correct_length(self):
        from audio.generator import generate_rhythmic_representation
        connections = [{"source": f"a{i}", "target": f"b{i}"} for i in range(3)]
        result = generate_rhythmic_representation([], connections)
        assert result.shape[0] == int(44100 * 5.0)


class TestGenerateAmbientRepresentation:
    def test_empty_inputs_returns_audio_with_drone(self):
        from audio.generator import generate_ambient_representation
        result = generate_ambient_representation([], [])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == int(44100 * 10.0)
        # Drone always present
        assert np.any(result != 0.0)

    def test_variables_add_harmonics(self):
        from audio.generator import generate_ambient_representation
        empty_result = generate_ambient_representation([], [])
        variables = [{"name": "s"}, {"name": "o"}]
        var_result = generate_ambient_representation(variables, [])
        # Adding variables changes the output
        assert not np.allclose(empty_result, var_result)

    def test_output_length_is_ten_seconds(self):
        from audio.generator import generate_ambient_representation
        result = generate_ambient_representation([{"name": "x"}], [])
        assert result.shape[0] == int(44100 * 10.0)


class TestGenerateSonificationAudio:
    def test_empty_dynamics_returns_silence(self):
        from audio.generator import generate_sonification_audio
        result = generate_sonification_audio([])
        assert isinstance(result, np.ndarray)
        assert np.all(result == 0.0)

    def test_single_dynamic_produces_audio(self):
        from audio.generator import generate_sonification_audio
        result = generate_sonification_audio([{"state": 0.5}])
        assert result.shape[0] > 0
        assert np.any(result != 0.0)

    def test_output_length_is_eight_seconds(self):
        from audio.generator import generate_sonification_audio
        result = generate_sonification_audio([{"x": 1.0}])
        assert result.shape[0] == int(44100 * 8.0)


class TestGenerateOscillatorAudio:
    @pytest.mark.parametrize("osc_type", ["sine", "square", "sawtooth", "triangle", "noise"])
    def test_oscillator_types_produce_audio(self, osc_type):
        from audio.generator import generate_oscillator_audio
        result = generate_oscillator_audio(440.0, 0.5, oscillator_type=osc_type)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == int(44100 * 0.5)

    def test_unknown_type_falls_back_gracefully(self):
        from audio.generator import generate_oscillator_audio
        result = generate_oscillator_audio(440.0, 0.5, oscillator_type="unknown")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0


class TestMixAudioChannels:
    def test_empty_channels_returns_empty(self):
        from audio.generator import mix_audio_channels
        result = mix_audio_channels([])
        assert len(result) == 0

    def test_single_channel_passthrough(self):
        from audio.generator import mix_audio_channels
        ch = np.ones(100)
        result = mix_audio_channels([ch])
        assert result.shape[0] == 100

    def test_add_mode_sums_channels(self):
        from audio.generator import mix_audio_channels
        a = np.ones(100)
        b = np.ones(100) * 2.0
        result = mix_audio_channels([a, b], mix_mode="add")
        assert np.allclose(result, 3.0)

    def test_average_mode(self):
        from audio.generator import mix_audio_channels
        a = np.ones(100)
        b = np.ones(100) * 3.0
        result = mix_audio_channels([a, b], mix_mode="average")
        assert np.allclose(result, 2.0)

    def test_unequal_length_channels_padded(self):
        from audio.generator import mix_audio_channels
        a = np.ones(200)
        b = np.ones(100)
        result = mix_audio_channels([a, b], mix_mode="add")
        assert result.shape[0] == 200


class TestSyntheticAudioGenerator:
    def test_generate_sine_wave(self):
        from audio.generator import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator()
        config = {"frequency": 440.0, "duration": 0.5, "oscillator_type": "sine", "sample_rate": 44100}
        result = gen.generate_synthetic_audio(config)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == int(44100 * 0.5)
        assert np.all(np.abs(result) <= 1.0 + 1e-6)

    def test_apply_adsr_envelope_changes_amplitude(self):
        from audio.generator import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator()
        audio = np.ones(4410)
        enveloped = gen.apply_envelope(audio, "ADSR")
        assert isinstance(enveloped, np.ndarray)
        assert enveloped.shape == audio.shape
        # Envelope should reduce start and end
        assert enveloped[0] < 0.5  # attack starts near 0
        assert enveloped[-1] < 0.5  # release ends near 0

    def test_supported_formats_present(self):
        from audio.generator import SyntheticAudioGenerator
        gen = SyntheticAudioGenerator()
        assert "wav" in gen.supported_formats
        assert "mp3" in gen.supported_formats
