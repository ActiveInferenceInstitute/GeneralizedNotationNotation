"""Tests for the canonical ``audio.sapf`` module surface.

The top-level SAPF alias package was removed (one home = ``src/gnn/audio/sapf/``).
These tests pin the canonical module's metadata and re-export surface directly.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gnn.audio.sapf as sapf


def test_sapf_import() -> None:
    """The canonical ``gnn.audio.sapf`` import must succeed."""
    assert sapf is not None


def test_sapf_metadata() -> None:
    """``audio.sapf`` exposes ``__version__`` and the single-source ``FEATURES`` dict."""
    assert hasattr(sapf, "__version__")
    assert hasattr(sapf, "FEATURES")
    assert isinstance(sapf.FEATURES, dict)
    # FEATURES is defined once, in module_info, and re-exported here.
    from gnn.audio.sapf.module_info import FEATURES as _features_source

    assert sapf.FEATURES is _features_source
    assert "gnn_to_sapf_conversion" in sapf.FEATURES
    assert "audio_generation" in sapf.FEATURES
    assert "sapf_validation" in sapf.FEATURES


def test_sapf_exported_functions() -> None:
    """The canonical module exports every public processing function."""
    expected_funcs: list[Any] = [
        "convert_gnn_to_sapf",
        "generate_sapf_audio",
        "generate_audio_from_sapf",
        "validate_sapf_code",
        "process_gnn_to_audio",
        "create_sapf_visualization",
        "generate_sapf_report",
    ]
    for func in expected_funcs:
        assert hasattr(sapf, func), f"Missing expected function: {func}"
        assert callable(getattr(sapf, func))


def test_sapf_get_module_info() -> None:
    """get_module_info returns the expected metadata shape."""
    info = sapf.get_module_info()
    assert isinstance(info, dict)
    assert "version" in info
    # audio.sapf declares SAPF as one of the supported formats.
    assert "supported_formats" in info
    formats_upper = [str(f).upper() for f in info["supported_formats"]]
    assert "SAPF" in formats_upper
