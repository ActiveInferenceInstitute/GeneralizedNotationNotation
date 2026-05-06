"""
Top-level ``sapf`` package — re-exports the SAPF implementation from
``audio.sapf`` so ``import sapf`` works at the top level.

This is documented in CLAUDE.md as an intentional composition shim, not a
fallback: SAPF lives under ``src/audio/sapf/`` because it's one modality of
the audio subsystem, but external callers and tests that reference it as a
peer module see it here.

Import resolution works under two invocation contexts:
  - ``PYTHONPATH=src python ...`` (standard)  → ``from audio import sapf``
  - ``python src/...`` (subprocess from pipeline) → ``from src.audio import sapf``

We handle both via a dynamic importlib call rather than hardcoding either
one, so generated framework scripts importing ``src`` don't crash with
``ModuleNotFoundError: audio`` when run as subprocesses.
"""

import importlib as _importlib


def _resolve_audio_sapf():
    """Resolve ``audio.sapf`` under either invocation context."""
    for candidate in ("audio.sapf", "src.audio.sapf"):
        try:
            return _importlib.import_module(candidate)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "Unable to resolve audio.sapf from either 'audio.sapf' or 'src.audio.sapf'"
    )


_audio_sapf = _resolve_audio_sapf()

convert_gnn_to_sapf = _audio_sapf.convert_gnn_to_sapf
create_sapf_visualization = _audio_sapf.create_sapf_visualization
generate_audio_from_sapf = _audio_sapf.generate_audio_from_sapf
generate_sapf_audio = _audio_sapf.generate_sapf_audio
generate_sapf_report = _audio_sapf.generate_sapf_report
process_gnn_to_audio = _audio_sapf.process_gnn_to_audio
validate_sapf_code = _audio_sapf.validate_sapf_code

__version__ = "1.6.0"
FEATURES = _audio_sapf.FEATURES


def get_module_info() -> dict:
    """Delegate to the underlying audio.sapf module metadata."""
    return _audio_sapf.get_module_info()


__all__ = [
    "convert_gnn_to_sapf",
    "generate_sapf_audio",
    "generate_audio_from_sapf",
    "validate_sapf_code",
    "process_gnn_to_audio",
    "create_sapf_visualization",
    "generate_sapf_report",
    "FEATURES",
    "get_module_info",
]
