"""Regression coverage for audio parsing, DSP edges, and WAV fallback."""

from __future__ import annotations

import builtins
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_audio_extractors_follow_real_gnn_sections_and_operators() -> None:
    from audio.processor import (
        extract_connections_for_audio,
        extract_variables_for_audio,
    )

    model = (
        REPOSITORY_ROOT / "input/gnn_files/discrete/actinf_pomdp_agent.md"
    ).read_text(encoding="utf-8")
    variables = extract_variables_for_audio(model)
    connections = extract_connections_for_audio(model)

    assert len(variables) == 13
    assert len({variable["name"] for variable in variables}) == 13
    assert {"A", "B", "s", "o", "π", "u"}.issubset(
        variable["name"] for variable in variables
    )
    assert len(connections) == 11
    assert {connection["operator"] for connection in connections} == {">", "-"}
    assert any(
        connection["source"] == "π" and connection["target"] == "u"
        for connection in connections
    )


def test_general_mixer_handles_unequal_stereo_mono_and_nonfinite_samples() -> None:
    from audio.generator import mix_audio_channels

    stereo = np.array([[1.0, np.nan], [np.inf, -np.inf], [0.5, -0.5]])
    mono = np.array([1.0])

    mixed = mix_audio_channels([stereo, mono], mix_mode="average")

    assert mixed.shape == (3, 2)
    assert np.isfinite(mixed).all()
    assert np.allclose(mixed[0], [1.0, 0.5])
    assert np.allclose(mixed[2], [0.25, -0.25])


def test_general_adsr_handles_empty_short_stereo_and_nan() -> None:
    from audio.generator import SyntheticAudioGenerator

    generator = SyntheticAudioGenerator()
    empty = generator.apply_envelope(np.empty((0, 2)), "ADSR")
    short = generator.apply_envelope(
        np.array([[np.nan, 1.0], [np.inf, -np.inf]]), "ADSR"
    )

    assert empty.shape == (0, 2)
    assert short.shape == (2, 2)
    assert np.isfinite(short).all()
    assert np.all(short[[0, -1]] == 0.0)


def test_sapf_adsr_handles_empty_and_nonfinite_stereo() -> None:
    from audio.sapf.audio_generators import apply_envelope

    empty = apply_envelope(np.empty((0, 2)), 0.1, 0.1, 0.7, 0.2, sample_rate=10)
    stereo = apply_envelope(
        np.array([[np.nan, 1.0], [0.5, np.inf], [-np.inf, -0.5]], dtype=float),
        1.0,
        1.0,
        0.7,
        1.0,
        sample_rate=10,
    )

    assert empty.shape == (0, 2)
    assert stereo.shape == (3, 2)
    assert np.isfinite(stereo).all()
    assert np.all(stereo[0] == 0.0)
    assert np.all(stereo[-1] == 0.0)


def test_sapf_mixer_preserves_stereo_and_validates_weights() -> None:
    from audio.sapf.audio_generators import mix_audio_channels

    stereo = np.array([[1.0, np.nan], [0.5, -0.5], [np.inf, -np.inf]])
    mono = np.array([1.0])

    mixed = mix_audio_channels([stereo, mono])

    assert mixed.shape == (3, 2)
    assert np.isfinite(mixed).all()
    with pytest.raises(ValueError, match="weights"):
        mix_audio_channels([stereo, mono], [1.0])
    with pytest.raises(ValueError, match="finite"):
        mix_audio_channels([stereo], [math.nan])


def test_sapf_oscillator_rejects_nonfinite_parameters() -> None:
    from audio.sapf.audio_generators import generate_oscillator_audio

    with pytest.raises(ValueError, match="frequency must be finite"):
        generate_oscillator_audio(math.nan, 0.5, 0.1)
    with pytest.raises(ValueError, match="duration must be non-negative"):
        generate_oscillator_audio(440.0, 0.5, -0.1)


def test_save_audio_uses_dependency_free_stereo_wav_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio.processor import save_audio_file

    original_import = builtins.__import__

    def import_without_soundfile(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "soundfile":
            raise ImportError("soundfile intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_soundfile)
    output_file = tmp_path / "nested" / "fallback.wav"
    samples = np.array([[0.0, np.nan], [np.inf, -np.inf], [0.25, -0.25]], dtype=float)

    save_audio_file(samples, output_file, sample_rate=8000)

    with wave.open(str(output_file), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == 8000
        assert wav_file.getnframes() == 3
        assert len(wav_file.readframes(3)) == 12


def test_basic_wav_supports_empty_stereo_and_rejects_invalid_layout(
    tmp_path: Path,
) -> None:
    from audio.processor import write_basic_wav

    empty_path = tmp_path / "empty-stereo.wav"
    write_basic_wav(np.empty((0, 2)), empty_path, 8000)
    with wave.open(str(empty_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getnframes() == 0

    with pytest.raises(ValueError, match="at least one channel"):
        write_basic_wav(np.empty((4, 0)), tmp_path / "zero-channel.wav", 8000)
    with pytest.raises(ValueError, match="real numbers"):
        write_basic_wav(np.array([1.0 + 2.0j]), tmp_path / "complex.wav", 8000)
