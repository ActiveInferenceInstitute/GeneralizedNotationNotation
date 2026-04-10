"""Tests for the SAPF audio bridge module."""



# Import the SAPF shim
try:
    import sapf
except ImportError:
    try:
        import src.sapf as sapf
    except ImportError:
        sapf = None

def test_sapf_import() -> None:
    """Verify that sapf can be imported."""
    assert sapf is not None

def test_sapf_metadata() -> None:
    """Verify that sapf re-exports expected attributes."""
    if sapf:
        assert hasattr(sapf, "__version__")
        assert hasattr(sapf, "FEATURES")
        assert isinstance(sapf.FEATURES, dict)
        assert "convert_gnn_to_sapf" in sapf.FEATURES

def test_sapf_exported_functions() -> None:
    """Verify that sapf re-exports core audio logic."""
    if sapf:
        expected_funcs = [
            "convert_gnn_to_sapf",
            "generate_sapf_audio",
            "validate_sapf_code",
            "process_gnn_to_audio",
        ]
        for func in expected_funcs:
            assert hasattr(sapf, func), f"Missing expected function: {func}"
            assert callable(getattr(sapf, func))

def test_sapf_get_module_info() -> None:
    """Test get_module_info returns expected data structure."""
    if sapf:
        info = sapf.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "supported_formats" in info
        assert "sapf" in info["supported_formats"]

def test_sapf_fallback_logic() -> None:
    """Test that get_module_info returns a minimal dict if delegation fails."""
    # We can't easily break the real audio.sapf import once it's imported,
    # but we can mock the _audio_sapf reference in the sapf module.
    if sapf:
        # We must not use mock/patch; manually replace the reference instead
        original = getattr(sapf, "_audio_sapf", None)
        sapf._audio_sapf = None
        try:
            info = sapf.get_module_info()
            assert info["version"] == sapf.__version__
            assert "SAPF audio bridge" in info["description"]
        finally:
            sapf._audio_sapf = original
