#!/usr/bin/env python3
"""
Test Unit Overall Tests

Real unit tests for core GNN utilities and helper functions.
No mocks - tests real implementations.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestPathUtilities:
    """Unit tests for path handling utilities."""

    def test_project_root_detection(self):
        """Test that project root can be determined."""
        from pathlib import Path
        
        # The test file should be within the project structure
        test_path = Path(__file__)
        
        # Navigate up to find project root (contains pyproject.toml)
        current = test_path.parent
        found_root = False
        for _ in range(5):  # Don't go too far up
            if (current / "pyproject.toml").exists():
                found_root = True
                break
            current = current.parent
        
        assert found_root, "Should find project root with pyproject.toml"

    def test_src_directory_exists(self):
        """Test that src directory exists and contains modules."""
        from pathlib import Path
        
        src_dir = Path(__file__).parent.parent
        
        assert src_dir.exists(), "src directory should exist"
        assert (src_dir / "__init__.py").exists(), "src should have __init__.py"


class TestModuleStructure:
    """Unit tests for module structure validation."""

    def test_gnn_module_exports(self):
        """Test that gnn module exports expected functions."""
        import gnn
        
        assert hasattr(gnn, 'parse_gnn_file'), "gnn should export parse_gnn_file"
        assert hasattr(gnn, 'validate_gnn_structure'), "gnn should export validate_gnn_structure"
        assert hasattr(gnn, '__version__'), "gnn should have __version__"

    def test_export_module_exports(self):
        """Test that export module exports expected functions."""
        import export
        
        assert hasattr(export, 'get_supported_formats'), "export should export get_supported_formats"
        assert hasattr(export, '__version__'), "export should have __version__"

    def test_render_module_exports(self):
        """Test that render module exports expected functions."""
        import render
        
        assert hasattr(render, 'process_render'), "render should export process_render"
        assert hasattr(render, 'get_available_renderers'), "render should export get_available_renderers"


class TestFeatureFlags:
    """Unit tests for feature flag consistency."""

    def test_gnn_features_dict(self):
        """Test GNN module has FEATURES dict."""
        import gnn
        
        assert hasattr(gnn, 'FEATURES'), "gnn should have FEATURES"
        assert isinstance(gnn.FEATURES, dict), "FEATURES should be a dict"
        assert len(gnn.FEATURES) > 0, "FEATURES should not be empty"

    def test_render_features_dict(self):
        """Test render module has FEATURES dict."""
        import render
        
        assert hasattr(render, 'FEATURES'), "render should have FEATURES"
        assert isinstance(render.FEATURES, dict), "FEATURES should be a dict"

    def test_src_package_version(self):
        """Test src package version matches expected format."""
        import src
        
        assert hasattr(src, '__version__'), "src should have __version__"
        version = src.__version__
        
        # Version should be in semver-like format
        parts = version.split('.')
        assert len(parts) >= 2, f"Version '{version}' should have at least major.minor"
