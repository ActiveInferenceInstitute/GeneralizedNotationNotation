#!/usr/bin/env python3
"""
Comprehensive Render Module Tests

This module tests the render module's ability to generate simulation code for various
target frameworks. All tests use real rendering functions with actual GNN specifications
and verify that output artifacts are correctly created.

Test Coverage:
- PyMDP code generation (test_render_to_pymdp, test_render_to_rxinfer_toml)
- DisCoPy categorical diagram rendering (test_render_to_discopy, test_render_to_discopy_combined)
- Multiple language/framework rendering (test_render_to_activeinference_jl, test_render_to_jax, test_render_to_jax_pomdp)

No mocking is used - all tests validate real render function execution.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


# Real-implementation render target tests
class TestRenderTargets:
    """
    Test rendering to different target frameworks.
    
    These tests verify that the GNN render module can successfully generate executable code
    for various target frameworks including PyMDP, RxInfer, DisCoPy, ActiveInference.jl, 
    and JAX. Each test uses real render function calls with actual GNN specifications 
    and verifies output artifacts are created correctly.
    
    Fixtures:
    - tmp_path: Temporary directory for artifact output
    - sample_gnn_spec: Basic GNN specification dict
    - mock_render_module: Real RealRenderModule instance (not a mock)
    """
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_pymdp(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to PyMDP format."""
        # Call real render function with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "pymdp", tmp_path)
        
        # Verify successful rendering
        assert ok is True, "PyMDP rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        
        # Verify artifacts are created
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_rxinfer_toml(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to RxInfer TOML format."""
        # Call real render function with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "rxinfer_toml", tmp_path)
        
        # Verify successful rendering
        assert ok is True, "RxInfer TOML rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        
        # Verify artifacts are created
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_discopy(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to DisCoPy format."""
        # Call real render function with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "discopy", tmp_path)
        
        # Verify successful rendering
        assert ok is True, "DisCoPy rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        
        # Verify artifacts are created
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_discopy_combined(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to DisCoPy combined format using real rendering."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "discopy_combined", tmp_path)
        
        assert ok is True, "DisCoPy combined rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"
        
        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_activeinference_jl(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to ActiveInference.jl format using real rendering."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "activeinference_jl", tmp_path)
        
        assert ok is True, "ActiveInference.jl rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"
        
        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_jax(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to JAX format using real rendering with content validation."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "jax", tmp_path)
        
        assert ok is True, "JAX rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"
        
        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"
        
        # Content validation: verify generated code has Active Inference constructs
        jax_artifacts = [a for a in artifacts if a.endswith('.py')]
        if jax_artifacts:
            content = (tmp_path / jax_artifacts[0]).read_text()
            assert 'import jax' in content or 'from jax' in content, \
                "Generated JAX code should import jax"
            assert 'create_params' in content or 'belief_update' in content, \
                "Generated JAX code should contain Active Inference functions"
            # Verify valid Python syntax
            compile(content, jax_artifacts[0], 'exec')
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_to_jax_pomdp(self, tmp_path, sample_gnn_spec, test_render_module):
        """Test rendering to JAX POMDP format using real rendering."""
        # Use real rendering call with actual data
        ok, msg, artifacts = test_render_module.render_gnn_spec(sample_gnn_spec, "jax_pomdp", tmp_path)
        
        assert ok is True, "JAX POMDP rendering should succeed"
        assert isinstance(msg, str), "Message should be string"
        assert isinstance(artifacts, list), "Artifacts should be a list"
        assert len(artifacts) > 0, "Should generate artifacts"
        
        # Verify artifacts are created in output directory
        for artifact in artifacts:
            artifact_path = tmp_path / artifact
            assert artifact_path.exists(), f"Artifact {artifact} should be created"


