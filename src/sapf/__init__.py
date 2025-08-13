"""
Top-level sapf package shim that re-exports real SAPF functionality from
`audio.sapf`. This keeps tests and external callers working with
`import sapf` without duplicating code.
"""
from audio.sapf import (
    convert_gnn_to_sapf,
    generate_sapf_audio,
    generate_audio_from_sapf,
    validate_sapf_code,
    process_gnn_to_audio,
    create_sapf_visualization,
    generate_sapf_report,
)

__all__ = [
    "convert_gnn_to_sapf",
    "generate_sapf_audio",
    "generate_audio_from_sapf",
    "validate_sapf_code",
    "process_gnn_to_audio",
    "create_sapf_visualization",
    "generate_sapf_report",
]


