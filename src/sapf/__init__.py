"""Top-level SAPF package surface backed by ``audio.sapf``."""

from typing import Any

from audio import sapf as _audio_sapf

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


__all__: list[Any] = [
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
