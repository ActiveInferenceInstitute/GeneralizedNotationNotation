"""Dependency-free NumPy audio effects for GNN sonification output.

Pure, deterministic DSP effects (no randomness, no filesystem, no logging)
matching the names advertised by :func:`audio.analyzer.get_audio_generation_options`
under the ``effects`` key: ``reverb``, ``delay``, ``chorus``, ``flanger``,
``distortion``, and ``filter``.

Each effect takes a mono or frames-by-channels float array and returns a
float64 array of the same shape (post-:func:`audio.validation.coerce_audio_array`
sanitization).  Effects never clip — they keep the output within roughly
``[-1, 1]`` where possible but downstream clipping is the caller's concern.
Use :func:`apply_effects_chain` to compose several effects by name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from .validation import coerce_audio_array, require_sample_rate

__all__ = [
    "EFFECTS",
    "apply_chorus",
    "apply_delay",
    "apply_distortion",
    "apply_effects_chain",
    "apply_flanger",
    "apply_highpass_filter",
    "apply_lowpass_filter",
    "apply_reverb",
]

EffectFn = Callable[..., np.ndarray]


def _mono(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (mono-or-first-channel array, original channel count)."""
    if audio.ndim == 1:
        return audio, 1
    return audio[:, 0], audio.shape[1]


def _broadcast(mono_signal: np.ndarray, channels: int) -> np.ndarray:
    return (
        np.repeat(mono_signal[:, np.newaxis], channels, axis=1)
        if channels > 1
        else mono_signal
    )


def apply_delay(
    audio: np.ndarray,
    sample_rate: int = 44100,
    time_s: float = 0.25,
    feedback: float = 0.3,
    mix: float = 0.3,
) -> np.ndarray:
    """Feedback delay (echo). ``feedback`` in [0, 1); ``mix`` blends dry/wet."""
    sr = require_sample_rate(sample_rate)
    samples = coerce_audio_array(audio)
    mono_signal, channels = _mono(samples)
    delay_samples = int(round(max(0.0, time_s) * sr))
    feedback = float(np.clip(feedback, 0.0, 0.999))
    mix = float(np.clip(mix, 0.0, 1.0))
    out = mono_signal.copy()
    if delay_samples > 0:
        delayed = np.zeros_like(mono_signal)
        for i in range(delay_samples, len(mono_signal)):
            delayed[i] = (
                mono_signal[i - delay_samples] + feedback * delayed[i - delay_samples]
            )
        out = (1.0 - mix) * mono_signal + mix * delayed
    return _broadcast(out, channels)


def apply_reverb(
    audio: np.ndarray,
    sample_rate: int = 44100,
    decay: float = 0.4,
    mix: float = 0.3,
) -> np.ndarray:
    """Lightweight multi-tap comb reverb approximation (no convolution)."""
    sr = require_sample_rate(sample_rate)
    samples = coerce_audio_array(audio)
    mono_signal, channels = _mono(samples)
    decay = float(np.clip(decay, 0.0, 0.95))
    mix = float(np.clip(mix, 0.0, 1.0))
    taps = (0.023, 0.029, 0.037, 0.041, 0.053)
    wet = np.zeros_like(mono_signal)
    for index, tap in enumerate(taps):
        offset = int(round(tap * sr))
        tap_decay = decay ** (index + 1)
        if offset <= 0:
            continue
        echoed = np.zeros_like(mono_signal)
        echoed[offset:] = mono_signal[:-offset]
        wet += tap_decay * echoed
    out = (1.0 - mix) * mono_signal + mix * wet
    return _broadcast(out, channels)


def _modulated_delay(
    audio: np.ndarray,
    sample_rate: int,
    depth_s: float,
    rate_hz: float,
    feedback: float,
    mix: float,
) -> np.ndarray:
    """Shared flanger/chorus modulation core: sine-modulated feedback delay."""
    sr = require_sample_rate(sample_rate)
    samples = coerce_audio_array(audio)
    mono_signal, channels = _mono(samples)
    depth_samples = max(0, int(round(depth_s * sr)))
    feedback = float(np.clip(feedback, 0.0, 0.95))
    mix = float(np.clip(mix, 0.0, 1.0))
    if depth_samples == 0:
        return samples
    n = len(mono_signal)
    t = np.arange(n) / sr
    modulation = (1.0 + np.sin(2.0 * np.pi * rate_hz * t)) * 0.5 * depth_samples
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        delay = modulation[i]
        low = int(np.floor(delay))
        frac = delay - low
        idx = i - low - 1
        if 0 <= idx:
            delayed = mono_signal[idx]
            if 0 <= idx - 1 < n:
                delayed += frac * (mono_signal[idx - 1] - mono_signal[idx])
            out[i] = (1.0 - mix) * mono_signal[i] + mix * (
                delayed + feedback * out[i - 1]
            )
        else:
            out[i] = mono_signal[i]
    return _broadcast(out, channels)


def apply_chorus(
    audio: np.ndarray,
    sample_rate: int = 44100,
    depth: float = 0.002,
    rate: float = 1.5,
    mix: float = 0.4,
) -> np.ndarray:
    """Sine-modulated delay chorus."""
    return _modulated_delay(audio, sample_rate, depth, rate, feedback=0.0, mix=mix)


def apply_flanger(
    audio: np.ndarray,
    sample_rate: int = 44100,
    depth: float = 0.001,
    rate: float = 0.5,
    mix: float = 0.5,
    feedback: float = 0.5,
) -> np.ndarray:
    """Feedback flanger (short modulated delay with regeneration)."""
    return _modulated_delay(audio, sample_rate, depth, rate, feedback=feedback, mix=mix)


def apply_distortion(audio: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """Tanh waveshaper distortion. ``drive`` >= 1 increases gain before shaping."""
    drive = float(max(1.0, drive))
    samples = coerce_audio_array(audio)
    shaped = np.tanh(drive * samples)
    return np.asarray(np.nan_to_num(shaped))


def apply_lowpass_filter(
    audio: np.ndarray, cutoff: float, sample_rate: int = 44100
) -> np.ndarray:
    """One-pole low-pass IIR filter (matches the SAPF reference implementation)."""
    sr = require_sample_rate(sample_rate)
    samples = coerce_audio_array(audio)
    mono_signal, channels = _mono(samples)
    cutoff_norm = float(np.clip(cutoff / (sr / 2.0), 0.001, 0.999))
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_norm)
    filtered = np.zeros_like(mono_signal)
    for i in range(1, len(mono_signal)):
        filtered[i] = alpha * mono_signal[i] + (1.0 - alpha) * filtered[i - 1]
    return _broadcast(filtered, channels)


def apply_highpass_filter(
    audio: np.ndarray, cutoff: float, sample_rate: int = 44100
) -> np.ndarray:
    """One-pole high-pass IIR filter (complement of the low-pass)."""
    sr = require_sample_rate(sample_rate)
    samples = coerce_audio_array(audio)
    mono_signal, channels = _mono(samples)
    cutoff_norm = float(np.clip(cutoff / (sr / 2.0), 0.001, 0.999))
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_norm)
    filtered = np.zeros_like(mono_signal)
    for i in range(1, len(mono_signal)):
        filtered[i] = (1.0 - alpha) * (
            filtered[i - 1] + mono_signal[i] - mono_signal[i - 1]
        )
    return _broadcast(filtered, channels)


def _filter_fn(**kwargs: Any) -> np.ndarray:
    cutoff = float(kwargs["cutoff"])
    kind = str(kwargs.get("kind", "lowpass")).lower()
    audio = kwargs["audio"]
    if kind.startswith("high"):
        return apply_highpass_filter(audio, cutoff, kwargs.get("sample_rate", 44100))
    return apply_lowpass_filter(audio, cutoff, kwargs.get("sample_rate", 44100))


EFFECTS: dict[str, EffectFn] = {
    "reverb": apply_reverb,
    "delay": apply_delay,
    "chorus": apply_chorus,
    "flanger": apply_flanger,
    "distortion": apply_distortion,
    "lowpass": apply_lowpass_filter,
    "highpass": apply_highpass_filter,
    "filter": _filter_fn,
}


def _resolve(effect: Any) -> tuple[EffectFn, dict[str, Any]]:
    """Turn a chain entry into (callable, kwargs) or raise ValueError."""
    if isinstance(effect, str):
        name = effect
        params: dict[str, Any] = {}
    elif isinstance(effect, Mapping):
        name = str(effect.get("type") or effect.get("name") or "")
        params = {k: v for k, v in effect.items() if k not in ("type", "name")}
    else:
        raise ValueError(
            f"effect entry must be a str or mapping, got {type(effect).__name__}"
        )
    fn = EFFECTS.get(name.lower())
    if fn is None:
        raise ValueError(f"unknown effect {name!r}; valid: {sorted(EFFECTS)}")
    return fn, params


def apply_effects_chain(
    audio: np.ndarray,
    effects: Sequence[Any],
    *,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Apply an ordered chain of effects to ``audio``.

    ``effects`` entries are either effect-name strings (``"reverb"``) or
    mappings (``{"type": "delay", "time_s": 0.3}``).  Unknown names raise
    ``ValueError`` listing the valid effects.  ``sample_rate`` is forwarded to
    effects that need it.  An empty chain returns sanitized ``audio`` unchanged.
    """
    samples = coerce_audio_array(audio)
    require_sample_rate(sample_rate)
    current = samples
    for entry in effects:
        fn, params = _resolve(entry)
        call_params = dict(params)
        call_params.setdefault("sample_rate", sample_rate)
        current = fn(current, **call_params)
    return np.asarray(np.nan_to_num(current))
