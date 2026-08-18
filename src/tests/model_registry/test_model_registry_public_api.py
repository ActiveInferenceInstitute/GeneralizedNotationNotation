"""Tests for model_registry module's public API surface not covered by existing tests.

Covers: process_model_registry, get_module_info, FEATURES, __version__.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModelRegistryConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import model_registry

        assert hasattr(model_registry, "FEATURES")
        assert isinstance(model_registry.FEATURES, dict)
        for key in (
            "model_versioning",
            "registry_management",
            "metadata_handling",
            "mcp_integration",
        ):
            assert key in model_registry.FEATURES

    def test_version(self) -> None:
        import model_registry

        assert hasattr(model_registry, "__version__")
        assert isinstance(model_registry.__version__, str)

    def test_get_module_info(self) -> None:
        from model_registry import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert info["name"] == "model_registry"
        assert "version" in info
        assert "description" in info
        assert "features" in info


class TestProcessModelRegistry:
    """Test process_model_registry function."""

    def test_process_empty_dir(self, tmp_path: Path) -> None:
        from model_registry import process_model_registry

        target = tmp_path / "empty"
        target.mkdir()
        out = tmp_path / "out"
        result = process_model_registry(target_dir=target, output_dir=out)
        assert isinstance(result, dict)
        # Should report 0 processed/registered files for empty dir
        assert "processed_files" in result
        assert result["processed_files"] == 0

    def test_process_nonexistent_dir(self, tmp_path: Path) -> None:
        from model_registry import process_model_registry

        out = tmp_path / "out"
        result = process_model_registry(
            target_dir=tmp_path / "nonexistent",
            output_dir=out,
        )
        assert isinstance(result, dict)
        assert "processed_files" in result

    def test_process_with_sample_file(self, sample_gnn_file: Any) -> None:
        from model_registry import process_model_registry

        target_dir = sample_gnn_file.parent
        output_dir = target_dir / "registry_out"
        result = process_model_registry(target_dir=target_dir, output_dir=output_dir)
        assert isinstance(result, dict)
        # Should have processed at least the sample file
        assert result["processed_files"] >= 1
        assert result["successful_registrations"] >= 1
        assert "registry_path" in result
        assert Path(result["registry_path"]).exists()

    def test_process_with_subdirs(self, sample_gnn_file: Any) -> None:
        from model_registry import process_model_registry

        target_dir = sample_gnn_file.parent
        output_dir = target_dir / "registry_recursive"
        result = process_model_registry(
            target_dir=target_dir, output_dir=output_dir, recursive=True
        )
        assert isinstance(result, dict)
        assert result["processed_files"] >= 1
