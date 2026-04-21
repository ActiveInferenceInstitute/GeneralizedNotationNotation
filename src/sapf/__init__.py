"""
Top-level ``sapf`` package — re-exports the SAPF implementation from
``audio.sapf`` so ``import sapf`` works at the top level.

This is documented in CLAUDE.md as an intentional composition shim, not a
fallback: SAPF lives under ``src/audio/sapf/`` because it's one modality of
the audio subsystem, but external callers and tests that reference it as a
peer module see it here. Phase 6: simplified to unconditional re-exports.
"""

from audio import sapf as _audio_sapf
from audio.sapf import (
    convert_gnn_to_sapf,
    create_sapf_visualization,
    generate_audio_from_sapf,
    generate_sapf_audio,
    generate_sapf_report,
    process_gnn_to_audio,
    validate_sapf_code,
)

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
