#!/usr/bin/env python3
"""
Test Export Overall Tests

This file contains comprehensive tests for the export module functionality.
"""

import pytest
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))



class TestExportModuleComprehensive:
    """Comprehensive tests for the export module."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_module_imports(self) -> None:
        """Test that export module can be imported."""
        try:
            import export
            assert hasattr(export, '__version__')
            assert hasattr(export, 'Exporter')
            assert hasattr(export, 'MultiFormatExporter')
            assert hasattr(export, 'get_supported_formats')
        except ImportError:
            pytest.skip("Export module not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_exporter_instantiation(self) -> None:
        """Test Exporter class instantiation."""
        try:
            from export import Exporter
            exporter = Exporter()
            assert exporter is not None
            assert hasattr(exporter, 'export_gnn_model')
            assert hasattr(exporter, 'validate_format')
        except ImportError:
            pytest.skip("Exporter not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_multi_format_exporter_instantiation(self) -> None:
        """Test MultiFormatExporter class instantiation."""
        try:
            from export import MultiFormatExporter
            exporter = MultiFormatExporter()
            assert exporter is not None
            assert hasattr(exporter, 'export_to_multiple_formats')
            assert hasattr(exporter, 'get_supported_formats')
        except ImportError:
            pytest.skip("MultiFormatExporter not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_module_info(self) -> None:
        """Test export module information retrieval."""
        try:
            from export import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'supported_formats' in info
        except ImportError:
            pytest.skip("Export module info not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_supported_formats(self) -> None:
        """Test supported formats retrieval."""
        try:
            from export import get_supported_formats
            formats = get_supported_formats()
            assert isinstance(formats, list)
            assert len(formats) > 0
            # Check for common formats
            expected_formats = ['json', 'xml', 'graphml', 'gexf', 'pickle']
            for fmt in expected_formats:
                assert fmt in formats
        except ImportError:
            pytest.skip("Export formats not available")


class TestExportFunctionality:
    """Tests for export functionality."""

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_gnn_model(self, comprehensive_test_data: Any) -> None:
        """Test GNN model export functionality."""
        try:
            from export import Exporter
            exporter = Exporter()

            # Test export with sample data
            gnn_data = comprehensive_test_data.get('gnn_content', 'test content')
            result = exporter.export_gnn_model(gnn_data, 'json')
            assert result is not None
        except ImportError:
            pytest.skip("Exporter not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_format_validation(self) -> None:
        """Test export format validation."""
        try:
            from export import validate_export_format
            result = validate_export_format('json')
            assert isinstance(result, bool)
            assert result is True

            result = validate_export_format('invalid_format')
            assert isinstance(result, bool)
            assert result is False
        except ImportError:
            pytest.skip("Export validation not available")

    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_multi_format_export(self, comprehensive_test_data: Any) -> None:
        """Test multi-format export functionality."""
        try:
            from export import MultiFormatExporter
            exporter = MultiFormatExporter()

            gnn_data = comprehensive_test_data.get('gnn_content', 'test content')
            formats = ['json', 'xml']
            result = exporter.export_to_multiple_formats(gnn_data, formats)
            assert result is not None
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("MultiFormatExporter not available")


class TestExportIntegration:
    """Integration tests for export module."""

    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_export_pipeline_integration(self, sample_gnn_files: Any, isolated_temp_dir: Any) -> None:
        """Test export module integration with pipeline."""
        try:
            from export import Exporter
            exporter = Exporter()

            # Test end-to-end export
            gnn_file = list(sample_gnn_files.values())[0]
            with open(gnn_file, 'r') as f:
                gnn_content = f.read()

            result = exporter.export_gnn_model(gnn_content, 'json')
            assert result is not None

        except ImportError:
            pytest.skip("Export module not available")

    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_export_mcp_integration(self) -> None:
        """Test export MCP integration."""
        try:
            from export.mcp import register_tools
            # Test that MCP tools can be registered
            assert callable(register_tools)
        except ImportError:
            pytest.skip("Export MCP not available")


def test_export_module_completeness() -> None:
    """Test that export module has all required components."""
    required_components = [
        'Exporter',
        'MultiFormatExporter',
        'get_module_info',
        'get_supported_formats',
        'validate_export_format'
    ]

    try:
        import export
        for component in required_components:
            assert hasattr(export, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Export module not available")


@pytest.mark.slow
def test_export_module_performance() -> None:
    """Test export module performance characteristics."""
    try:
        from export import Exporter
        import time

        exporter = Exporter()
        start_time = time.time()

        # Test export performance
        result = exporter.export_gnn_model("test content", 'json')

        processing_time = time.time() - start_time
        assert processing_time < 5.0  # Should complete within 5 seconds

    except ImportError:
        pytest.skip("Export module not available")



class TestExportUtils:
    """Smoke tests for export.utils sub-module."""

    def test_module_importable(self):
        from export import utils  # noqa: F401

    def test_get_module_info_returns_dict(self):
        from export.utils import get_module_info
        result = get_module_info()
        assert isinstance(result, dict)

    def test_get_supported_formats_returns_dict(self):
        from export.utils import get_supported_formats
        result = get_supported_formats()
        assert isinstance(result, dict)
        assert len(result) > 0
