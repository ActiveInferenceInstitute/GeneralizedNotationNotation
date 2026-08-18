"""Tests for ml_integration module's public API surface not covered by existing tests.

Covers: check_ml_frameworks (extended), get_module_info, FEATURES, __version__.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMLIntegrationConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import ml_integration

        assert hasattr(ml_integration, "FEATURES")
        assert isinstance(ml_integration.FEATURES, dict)
        for key in (
            "model_training",
            "model_inference",
            "pipeline_integration",
            "mcp_integration",
        ):
            assert key in ml_integration.FEATURES

    def test_version(self) -> None:
        import ml_integration

        assert hasattr(ml_integration, "__version__")
        assert isinstance(ml_integration.__version__, str)

    def test_get_module_info(self) -> None:
        from ml_integration import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert info["name"] == "ml_integration"
        assert "version" in info
        assert "description" in info
        assert "features" in info


class TestCheckMLFrameworksExtended:
    """Extended tests for check_ml_frameworks."""

    def test_check_returns_pytorch_or_torch_key(self) -> None:
        from ml_integration import check_ml_frameworks

        result = check_ml_frameworks()
        assert isinstance(result, dict)
        # Should have either 'pytorch' or 'torch' key
        has_torch = "pytorch" in result or "torch" in result
        assert has_torch, f"Expected pytorch or torch key, got {list(result.keys())}"

    def test_check_returns_sklearn(self) -> None:
        from ml_integration import check_ml_frameworks

        result = check_ml_frameworks()
        assert "sklearn" in result
        assert isinstance(result["sklearn"], dict)
        assert "available" in result["sklearn"]
        assert "version" in result["sklearn"]

    def test_check_framework_values_are_dicts(self) -> None:
        from ml_integration import check_ml_frameworks

        result = check_ml_frameworks()
        for fw, val in result.items():
            assert isinstance(val, dict), f"Framework {fw} value is not dict"

    def test_check_tensorflow_present(self) -> None:
        from ml_integration import check_ml_frameworks

        result = check_ml_frameworks()
        assert "tensorflow" in result
        assert isinstance(result["tensorflow"], dict)

    def test_check_jax_present(self) -> None:
        from ml_integration import check_ml_frameworks

        result = check_ml_frameworks()
        assert "jax" in result
        assert isinstance(result["jax"], dict)
