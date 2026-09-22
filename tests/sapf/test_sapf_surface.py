"""Golden export-surface contract for ``gnn.audio.sapf``.

Pins the public surface declared in ``src/gnn/audio/sapf/__init__.py``: every
``__all__`` entry must resolve, and the metadata names (FEATURES,
get_module_info) must be the single-source objects from ``module_info``.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.fast


def test_audio_sapf_exports_resolve() -> None:
    """Every ``gnn.audio.sapf.__all__`` entry resolves — guards the canonical SAPF surface."""
    audio_sapf = importlib.import_module("gnn.audio.sapf")
    for name in audio_sapf.__all__:
        assert hasattr(audio_sapf, name), f"gnn.audio.sapf.{name} missing"


def test_audio_sapf_metadata_is_single_sourced() -> None:
    """FEATURES and get_module_info live only in module_info; __init__ re-exports them."""
    audio_sapf = importlib.import_module("gnn.audio.sapf")
    module_info = importlib.import_module("gnn.audio.sapf.module_info")

    assert audio_sapf.FEATURES is module_info.FEATURES
    assert audio_sapf.get_module_info is module_info.get_module_info

    # No other module in the package redefines FEATURES.
    import gnn.audio.sapf.audio_generators as audio_generators
    import gnn.audio.sapf.processor as processor
    import gnn.audio.sapf.sapf_gnn_processor as sapf_gnn_processor

    for module in (audio_generators, processor, sapf_gnn_processor):
        assert not hasattr(module, "FEATURES"), f"{module.__name__} redefines FEATURES"


def test_audio_sapf_all_names_are_public_processing_surface() -> None:
    """The golden __all__ inventory: processing, audio generation, utilities, metadata."""
    audio_sapf = importlib.import_module("gnn.audio.sapf")
    assert set(audio_sapf.__all__) == {
        # Core SAPF processing
        "SAPFGNNProcessor",
        "convert_gnn_to_sapf",
        "generate_audio_from_sapf",
        "validate_sapf_code",
        # Processor functions
        "process_gnn_to_audio",
        "generate_sapf_audio",
        "create_sapf_visualization",
        "generate_sapf_report",
        # Audio generation
        "SyntheticAudioGenerator",
        "generate_oscillator_audio",
        "apply_envelope",
        "mix_audio_channels",
        # Utility functions
        "get_module_info",
        "get_audio_generation_options",
        "register_tools",
        # Metadata
        "FEATURES",
        "__version__",
    }
