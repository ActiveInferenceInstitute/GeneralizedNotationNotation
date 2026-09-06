"""Shared validation primitives for the audio package.

Single source of truth for the sample-array and sample-rate coercion that
previously lived in three near-identical private copies
(``audio.generator._audio_array`` / ``audio.processor._clean_audio_array`` /
``audio.sapf.audio_generators._audio_array`` and friends).

All functions are pure: no logging, no filesystem access, no global state.
Error messages are part of the contract — existing tests pin the substrings
``"real numbers"``, ``"numeric"``, ``"at least one channel"``, and
``"sample_rate must be a positive integer"``.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

__all__ = [
    "coerce_audio_array",
    "coerce_finite",
    "coerce_sample_rate",
    "require_finite",
    "require_sample_rate",
]


def coerce_finite(value: Any, default: float) -> float:
    """Return ``value`` as a finite float, or ``default`` when not coercible.

    Used where a bad knob should degrade to a default rather than raise.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def require_finite(value: Any, name: str) -> float:
    """Return ``value`` as a finite float or raise ``ValueError``.

    Error messages embed ``name`` (e.g. ``"frequency must be finite"``) so
    callers can pass parameter names through.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def coerce_sample_rate(value: Any, default: int = 44100) -> int:
    """Return ``value`` as a positive integer sample rate, else ``default``.

    Booleans are rejected (``True`` is not a sample rate).
    """
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return parsed if parsed > 0 and parsed == value else default


def require_sample_rate(value: Any) -> int:
    """Return ``value`` as a positive integer sample rate or raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError("sample_rate must be a positive integer")
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError("sample_rate must be a positive integer")
    return parsed


def coerce_audio_array(audio: Any, name: str = "audio") -> np.ndarray:
    """Return finite mono or frames-by-channels float64 samples.

    Rejects complex input, non-numeric input, anything but 1-D or 2-D arrays,
    and zero-channel frames.  ``NaN``/``±inf`` samples are replaced with
    ``0.0``/``±1.0`` respectively.
    """
    raw = np.asarray(audio)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} samples must be real numbers")
    try:
        samples = np.asarray(audio, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} samples must be numeric") from exc
    if samples.ndim not in (1, 2):
        raise ValueError(f"{name} must be mono or frames-by-channels")
    if samples.ndim == 2 and samples.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one channel")
    return np.asarray(np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0))
