#!/usr/bin/env python3
"""Minimal test configuration for GNN pipeline tests."""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

# Test markers
pytestmark = []

@pytest.fixture
def isolated_temp_dir():
    """Provide isolated temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)

@pytest.fixture
def sample_gnn_files():
    """Provide sample GNN files for testing."""
    return {}

# Minimal configuration to avoid pytest issues
def pytest_configure(config):
    """Configure pytest."""
    pass

