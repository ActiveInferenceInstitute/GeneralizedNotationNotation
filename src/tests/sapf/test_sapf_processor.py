"""Tests for the ``sapf`` top-level re-export package.

Phase 6: the shim was simplified to unconditionally delegate to
``audio.sapf``. FEATURES and get_module_info now pass through the real
module's shape directly — these tests verify the delegation is wired up.
"""

import sapf


def test_sapf_import() -> None:
    """The top-level ``sapf`` import must succeed."""
    assert sapf is not None


def test_sapf_metadata() -> None:
    """``sapf`` re-exports ``__version__`` and ``FEATURES`` from the audio subsystem."""
    assert hasattr(sapf, "__version__")
    assert hasattr(sapf, "FEATURES")
    assert isinstance(sapf.FEATURES, dict)
    # Actual key inventory comes from audio.sapf (the real implementation).
    assert "gnn_to_sapf_conversion" in sapf.FEATURES
    assert "audio_generation" in sapf.FEATURES
    assert "sapf_validation" in sapf.FEATURES


def test_sapf_exported_functions() -> None:
    """The shim re-exports every public function from audio.sapf."""
    expected_funcs = [
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
    """get_module_info delegates to audio.sapf and returns expected shape."""
    info = sapf.get_module_info()
    assert isinstance(info, dict)
    assert "version" in info
    # audio.sapf declares SAPF as one of the supported formats.
    assert "supported_formats" in info
    formats_upper = [str(f).upper() for f in info["supported_formats"]]
    assert "SAPF" in formats_upper


def test_sapf_delegates_to_audio_sapf_for_metadata() -> None:
    """Phase 6: verify the shim keeps a live reference to audio.sapf rather
    than maintaining its own duplicate metadata."""
    from audio import sapf as audio_sapf_direct
    # The shim's FEATURES must be the same object as audio.sapf's.
    assert sapf.FEATURES is audio_sapf_direct.FEATURES
