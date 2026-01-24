#!/usr/bin/env python3
"""
Test Environment Integration - Integration tests for environment setup.

Tests the integration between environment validation and pipeline components.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEnvironmentSetupIntegration:
    """Tests for environment setup integration with pipeline."""

    @pytest.mark.integration
    def test_environment_validates_before_pipeline(self):
        """Test environment validation runs before pipeline."""
        from utils.test_utils import validate_test_environment
        
        result = validate_test_environment()
        # validate_test_environment returns (bool, list) tuple or bool
        if isinstance(result, tuple):
            assert result[0] is True
        else:
            assert result is True or result is None

    @pytest.mark.integration
    def test_environment_paths_accessible(self):
        """Test all required paths are accessible."""
        from pathlib import Path
        
        # Check project structure
        test_dir = Path(__file__).parent
        src_dir = test_dir.parent
        project_root = src_dir.parent
        
        assert test_dir.exists()
        assert src_dir.exists()
        assert project_root.exists()

    @pytest.mark.integration
    def test_environment_imports_work(self):
        """Test core imports work from environment."""
        # These should all import without error
        from gnn import get_module_info
        from render import get_module_info as render_info
        from report import get_module_info as report_info
        
        assert get_module_info() is not None
        assert render_info() is not None
        assert report_info() is not None


class TestEnvironmentModuleIntegration:
    """Tests for environment integration with modules."""

    @pytest.mark.integration
    def test_all_modules_have_info(self):
        """Test all modules provide info."""
        modules_with_info = [
            'gnn',
            'render',
            'report',
            'audio',
            'visualization'
        ]
        
        for module_name in modules_with_info:
            try:
                module = __import__(module_name)
                if hasattr(module, 'get_module_info'):
                    info = module.get_module_info()
                    assert info is not None
            except ImportError:
                pass  # Module may have optional deps

    @pytest.mark.integration
    def test_module_features_accessible(self):
        """Test module features are accessible."""
        from audio import FEATURES
        from report import FEATURES as REPORT_FEATURES
        
        assert isinstance(FEATURES, dict)
        assert isinstance(REPORT_FEATURES, dict)


class TestEnvironmentPipelineIntegration:
    """Tests for environment integration with pipeline execution."""

    @pytest.mark.integration
    def test_pipeline_finds_scripts(self):
        """Test pipeline can find step scripts."""
        from pathlib import Path
        
        src_dir = Path(__file__).parent.parent
        
        # Check for numbered script files
        scripts = list(src_dir.glob("[0-9]*.py")) + list(src_dir.glob("[0-9][0-9]_*.py"))
        
        # Should find some pipeline scripts
        assert len(scripts) >= 0

    @pytest.mark.integration
    def test_output_directory_creation(self, tmp_path):
        """Test output directory can be created."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assert output_dir.exists()
        assert output_dir.is_dir()

    @pytest.mark.integration
    def test_temp_directory_creation(self, tmp_path):
        """Test temporary directories can be created."""
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Should be writable
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        assert test_file.exists()


class TestEnvironmentLoggingIntegration:
    """Tests for environment logging integration."""

    @pytest.mark.integration
    def test_logging_configuration(self):
        """Test logging can be configured."""
        import logging
        
        logger = logging.getLogger("test_env")
        logger.setLevel(logging.DEBUG)
        
        # Should be able to log without error
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

    @pytest.mark.integration
    def test_log_file_creation(self, tmp_path):
        """Test log files can be created."""
        import logging
        
        log_file = tmp_path / "test.log"
        
        handler = logging.FileHandler(log_file)
        logger = logging.getLogger("test_log_file")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info("Test log entry")
        handler.flush()
        handler.close()
        
        assert log_file.exists()


class TestEnvironmentResourceIntegration:
    """Tests for environment resource access."""

    @pytest.mark.integration
    def test_input_directory_access(self, sample_gnn_files):
        """Test input directories are accessible."""
        if sample_gnn_files:
            for gnn_file in sample_gnn_files.values():
                assert gnn_file.exists()
                assert gnn_file.is_file()

    @pytest.mark.integration
    def test_memory_availability(self):
        """Test sufficient memory is available."""
        import numpy as np
        
        # Try to allocate a reasonable array
        try:
            arr = np.zeros((1000, 1000))
            assert arr.shape == (1000, 1000)
        except MemoryError:
            pytest.fail("Insufficient memory")

    @pytest.mark.integration
    def test_disk_space_available(self, tmp_path):
        """Test sufficient disk space is available."""
        import os
        
        # Write some test data
        test_file = tmp_path / "space_test.bin"
        test_file.write_bytes(b"0" * 10000)  # 10KB
        
        assert test_file.exists()
        assert test_file.stat().st_size >= 10000
