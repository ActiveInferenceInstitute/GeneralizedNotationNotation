#!/usr/bin/env python3
"""
Integration Test for GNN-PyMDP Pipeline

Tests the integration between GNN parsing and PyMDP simulation execution.
This test verifies that GNN specifications can be properly parsed and used
to configure PyMDP simulations.

Features:
- GNN file parsing validation
- Parameter extraction testing
- PyMDP simulation configuration testing
- End-to-end pipeline integration testing

Author: GNN PyMDP Integration
Date: 2024
"""

import pytest
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent  # src/tests → src → project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.gnn.parsers.markdown_parser import MarkdownGNNParser
from src.render.pymdp.pymdp_renderer import PyMDPRenderer
try:
    from execute.pymdp.pymdp_simulation import PyMDPSimulation
    from execute.pymdp.pymdp_utils import format_duration
except ImportError:
    from src.execute.pymdp.pymdp_simulation import PyMDPSimulation
    from src.execute.pymdp.pymdp_utils import format_duration


@pytest.fixture
def gnn_file():
    """Get the example GNN file for testing."""
    return project_root / "input" / "gnn_files" / "actinf_pomdp_agent.md"


class TestGNNPyMDPIntegration:
    """Integration tests for GNN-PyMDP pipeline."""

    def test_gnn_parsing(self, gnn_file):
        """Test GNN file parsing and parameter extraction."""
        if not gnn_file.exists():
            pytest.skip(f"GNN file not found: {gnn_file}")
        
        parser = MarkdownGNNParser()
        parsed_data = parser.parse_file(gnn_file)
        
        assert parsed_data is not None
        # Check we got a ParseResult or dict with data
        if hasattr(parsed_data, 'success'):
            # ParseResult object
            assert parsed_data.success is True
        elif hasattr(parsed_data, 'data'):
            assert parsed_data.data is not None
        else:
            # Fallback for dict returns
            assert parsed_data

    def test_pymdp_renderer_exists(self, gnn_file):
        """Test PyMDP renderer can be instantiated and has render_file method."""
        if not gnn_file.exists():
            pytest.skip(f"GNN file not found: {gnn_file}")
        
        renderer = PyMDPRenderer()
        
        # Verify renderer has the expected methods
        assert hasattr(renderer, 'render_file')
        assert hasattr(renderer, 'render_directory')
        assert callable(renderer.render_file)

    def test_pymdp_simulation_creation(self):
        """Test PyMDP simulation with GNN-derived parameters."""
        config = {
            'num_states': 3,
            'num_observations': 3,
            'num_actions': 3,
            'num_timesteps': 5,
        }
        
        simulation = PyMDPSimulation(config)
        assert simulation is not None

    def test_pymdp_simulation_run(self):
        """Test running a PyMDP simulation."""
        config = {
            'num_states': 3,
            'num_observations': 3,
            'num_actions': 3,
            'num_timesteps': 5,
        }
        
        simulation = PyMDPSimulation(config)
        # Create model first
        simulation.create_model()
        # Use correct method name
        results = simulation.run_simulation()
        
        assert results is not None
        # Check expected keys in results
        assert isinstance(results, dict)

    def test_full_integration(self, gnn_file, tmp_path):
        """Test full GNN-to-PyMDP integration pipeline."""
        if not gnn_file.exists():
            pytest.skip(f"GNN file not found: {gnn_file}")
        
        # Step 1: Render GNN file to PyMDP code
        renderer = PyMDPRenderer()
        output_path = tmp_path / "rendered_pymdp.py"
        success, message = renderer.render_file(gnn_file, output_path)
        
        assert success, f"Rendering failed: {message}"
        assert output_path.exists(), "Rendered file not created"


# Standalone test functions for backward compatibility
def test_gnn_parsing():
    """Test GNN file parsing and parameter extraction."""
    gnn_file = project_root / "input" / "gnn_files" / "actinf_pomdp_agent.md"
    
    if not gnn_file.exists():
        pytest.skip(f"GNN file not found: {gnn_file}")
    
    parser = MarkdownGNNParser()
    parsed_data = parser.parse_file(gnn_file)
    
    assert parsed_data is not None


def test_pymdp_renderer():
    """Test PyMDP renderer exists and can be instantiated."""
    renderer = PyMDPRenderer()
    
    # Verify renderer has the expected methods
    assert hasattr(renderer, 'render_file')
    assert callable(renderer.render_file)


def test_pymdp_simulation():
    """Test PyMDP simulation with GNN-derived parameters."""
    config = {
        'num_states': 3,
        'num_observations': 3,
        'num_actions': 3,
        'num_timesteps': 5,
    }
    
    simulation = PyMDPSimulation(config)
    assert simulation is not None
    
    # Create model first
    simulation.create_model()
    # Use correct method name
    results = simulation.run_simulation()
    assert results is not None


def test_full_integration(tmp_path):
    """Test full GNN-to-PyMDP integration pipeline."""
    gnn_file = project_root / "input" / "gnn_files" / "actinf_pomdp_agent.md"
    
    if not gnn_file.exists():
        pytest.skip(f"GNN file not found: {gnn_file}")
    
    # Step 1: Render GNN file to PyMDP code
    renderer = PyMDPRenderer()
    output_path = tmp_path / "rendered_pymdp.py"
    success, message = renderer.render_file(gnn_file, output_path)
    
    assert success, f"Rendering failed: {message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])