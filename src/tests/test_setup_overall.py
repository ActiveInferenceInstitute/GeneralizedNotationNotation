"""
Test suite for Setup module.

Tests environment setup, UV integration, and dependency management.
"""

import pytest
from pathlib import Path


class TestSetupModule:
    """Test suite for Setup module functionality."""

    def test_module_imports(self):
        """Test that setup module can be imported."""
        from setup import (
            setup_uv_environment,
            validate_uv_setup,
            check_uv_availability,
            FEATURES,
            __version__
        )
        assert __version__ is not None
        assert isinstance(FEATURES, dict)
        assert callable(setup_uv_environment)
        assert callable(validate_uv_setup)
        assert callable(check_uv_availability)

    def test_features_available(self):
        """Test that FEATURES dict is properly populated."""
        from setup import FEATURES

        expected_features = [
            'uv_environment_setup',
            'uv_dependency_management',
            'system_validation',
            'mcp_integration'
        ]

        for feature in expected_features:
            assert feature in FEATURES, f"Missing feature: {feature}"

    def test_version_format(self):
        """Test version string format."""
        from setup import __version__

        # Should be semantic versioning format
        parts = __version__.split('.')
        assert len(parts) >= 2, "Version should have at least major.minor"
        assert all(p.isdigit() for p in parts[:2]), "Major and minor should be numeric"

    def test_check_uv_availability(self):
        """Test UV availability check."""
        from setup import check_uv_availability

        result = check_uv_availability()
        assert isinstance(result, bool)

    def test_validate_uv_setup(self):
        """Test UV setup validation."""
        from setup import validate_uv_setup

        result = validate_uv_setup()
        assert isinstance(result, dict)
        assert 'overall_status' in result or 'valid' in result or isinstance(result.get('uv_available'), bool)

    def test_environment_manager_class(self):
        """Test EnvironmentManager class exists and works."""
        from setup import EnvironmentManager

        manager = EnvironmentManager()
        assert hasattr(manager, 'setup_environment')
        assert hasattr(manager, 'validate_environment')

        # Test methods are callable
        assert callable(manager.setup_environment)
        assert callable(manager.validate_environment)

    def test_virtual_environment_class(self):
        """Test VirtualEnvironment class exists."""
        from setup import VirtualEnvironment

        venv = VirtualEnvironment("test_env")
        assert venv.name == "test_env"
        assert hasattr(venv, 'create')
        assert hasattr(venv, 'activate')

    def test_get_module_info(self):
        """Test get_module_info function."""
        from setup import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        # Should have environment_types per the module definition
        assert 'environment_types' in info

    def test_check_python_version(self):
        """Test Python version check."""
        from setup import check_python_version

        result = check_python_version()
        assert result is True  # We're running Python 3+

    def test_optional_groups_constant(self):
        """Test OPTIONAL_GROUPS constant exists."""
        from setup import OPTIONAL_GROUPS

        assert isinstance(OPTIONAL_GROUPS, (dict, list, tuple))


class TestSetupUtilities:
    """Test setup utility functions."""

    def test_ensure_directory(self, safe_filesystem):
        """Test directory creation utility."""
        from setup import ensure_directory

        test_dir = safe_filesystem.temp_dir / "test_ensure_dir"
        ensure_directory(test_dir)
        assert test_dir.exists()

    def test_find_gnn_files(self, safe_filesystem):
        """Test GNN file discovery."""
        from setup import find_gnn_files

        # Create test GNN file
        gnn_content = """# Test Model
## StateSpaceBlock
s[3]
"""
        test_file = safe_filesystem.create_file("test.md", gnn_content)

        files = find_gnn_files(safe_filesystem.temp_dir)
        assert isinstance(files, list)

    def test_get_output_paths(self, safe_filesystem):
        """Test output path generation."""
        from setup import get_output_paths

        # get_output_paths takes only base_output_dir parameter
        paths = get_output_paths(safe_filesystem.temp_dir)
        assert isinstance(paths, dict)


class TestSetupIntegration:
    """Integration tests for setup module."""

    def test_setup_environment_function(self):
        """Test setup_environment utility."""
        from setup import setup_environment

        # Should return success (True) or dict with status
        result = setup_environment()
        assert result is not None

    def test_install_dependencies_function(self):
        """Test install_dependencies utility."""
        from setup import install_dependencies

        # Should be callable and not crash
        assert callable(install_dependencies)
