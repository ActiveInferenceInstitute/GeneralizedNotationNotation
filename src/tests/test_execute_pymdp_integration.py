#!/usr/bin/env python3
"""
Real PyMDP execution tests.

This module tests actual PyMDP functionality with real simulations.
Tests gracefully skip if PyMDP is not installed.
"""

import pytest
from pathlib import Path
import sys
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if PyMDP is available
try:
    from pymdp.agent import Agent
    from pymdp import utils
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False

try:
    from execute.pymdp.simple_simulation import run_simple_pymdp_simulation
    from execute.pymdp.package_detector import (
        detect_pymdp_installation,
        is_correct_pymdp_package,
        validate_pymdp_for_execution
    )
    EXECUTE_MODULE_AVAILABLE = True
except ImportError as e:
    EXECUTE_MODULE_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not PYMDP_AVAILABLE, reason="PyMDP (inferactively-pymdp) not installed")
@pytest.mark.skipif(not EXECUTE_MODULE_AVAILABLE, reason="Execute module not available")
class TestPyMDPRealExecution:
    """Test real PyMDP execution with actual simulations."""
    
    def test_pymdp_agent_import(self):
        """Test that PyMDP Agent can be imported using modern API."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        assert Agent is not None
        assert utils is not None
        assert callable(Agent)
    
    def test_pymdp_agent_creation(self):
        """Test creating a PyMDP Agent with minimal matrices."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        # Create minimal matrices for a 2-state, 2-observation, 1-action model
        num_states = 2
        num_obs = 2
        num_actions = 1
        
        # A matrix: observation model (num_obs x num_states)
        A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        A_obj = utils.obj_array(1)
        A_obj[0] = A
        
        # B matrix: transition model (num_states x num_states x num_actions)
        B = np.zeros((num_states, num_states, num_actions), dtype=np.float64)
        B[:, :, 0] = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
        B_obj = utils.obj_array(1)
        B_obj[0] = B
        
        # C vector: preferences (num_obs)
        C = np.array([0.0, 0.0], dtype=np.float64)
        C_obj = utils.obj_array(1)
        C_obj[0] = C
        
        # D vector: prior over states (num_states)
        D = np.array([0.5, 0.5], dtype=np.float64)
        D_obj = utils.obj_array(1)
        D_obj[0] = D
        
        # Create agent
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
        
        assert agent is not None
        assert hasattr(agent, 'infer_states')
        assert hasattr(agent, 'infer_policies')
        assert hasattr(agent, 'sample_action')
    
    def test_pymdp_simple_simulation_execution(self, tmp_path):
        """Test running a simple PyMDP simulation."""
        # Create minimal GNN spec
        gnn_spec = {
            "model_name": "test_pymdp_model",
            "initialparameterization": {
                "A": [[0.9, 0.1], [0.1, 0.9]],
                # B matrix: shape (next_state x prev_state x num_actions) = (2, 2, 1)
                # Each column (axis 0) sums to 1.0
                "B": [[[0.8, 0.2], [0.2, 0.8]]],  # shape (1, 2, 2) - will be transposed to (2, 2, 1) by simple_simulation
                "C": [0.0, 0.0],
                "D": [0.5, 0.5]
            }
        }
        
        output_dir = tmp_path / "pymdp_output"
        output_dir.mkdir()
        
        success, results = run_simple_pymdp_simulation(gnn_spec, output_dir)
        
        assert success is True
        assert isinstance(results, dict)
        assert "success" in results
        assert results["success"] is True
        assert "framework" in results
        assert results["framework"] == "PyMDP"
        assert "observations" in results
        assert "actions" in results
        assert "beliefs" in results
    
    def test_pymdp_package_detection_with_real_installation(self):
        """Test that package detector correctly identifies installed PyMDP."""
        detection = detect_pymdp_installation()
        
        assert detection["installed"] is True
        assert detection["correct_package"] is True
        assert detection["has_agent"] is True
        assert detection["wrong_package"] is False
    
    def test_is_correct_pymdp_package_with_real_installation(self):
        """Test correct package detection with real installation."""
        result = is_correct_pymdp_package()
        assert result is True
    
    def test_validate_pymdp_for_execution_with_real_installation(self):
        """Test validation with real PyMDP installation."""
        validation = validate_pymdp_for_execution()
        
        assert validation["ready"] is True
        assert validation["detection"]["correct_package"] is True
        assert validation["can_auto_install"] is True


@pytest.mark.skipif(not EXECUTE_MODULE_AVAILABLE, reason="Execute module not available")
class TestPyMDPErrorHandling:
    """Test PyMDP error handling when package is wrong or missing."""
    
    def test_package_detection_structure(self):
        """Test that package detection returns expected structure even when not installed."""
        detection = detect_pymdp_installation()
        
        assert isinstance(detection, dict)
        assert "installed" in detection
        assert "correct_package" in detection
        assert "wrong_package" in detection
        assert isinstance(detection["installed"], bool)
    
    def test_validation_structure(self):
        """Test that validation returns expected structure."""
        validation = validate_pymdp_for_execution()
        
        assert isinstance(validation, dict)
        assert "ready" in validation
        assert "detection" in validation
        assert "instructions" in validation
        assert isinstance(validation["ready"], bool)


@pytest.mark.skipif(not PYMDP_AVAILABLE, reason="PyMDP (inferactively-pymdp) not installed")
@pytest.mark.skipif(not EXECUTE_MODULE_AVAILABLE, reason="Execute module not available")
class TestPyMDPModernAPI:
    """Test that we're using the modern PyMDP API correctly."""
    
    def test_modern_import_works(self):
        """Test that modern import pattern works."""
        # This should work with inferactively-pymdp
        from pymdp.agent import Agent
        from pymdp import utils
        
        assert Agent is not None
        assert utils is not None
    
    def test_agent_has_required_methods(self):
        """Test that Agent has all required methods for simulation."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        # Create minimal agent
        A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        B = np.zeros((2, 2, 1), dtype=np.float64)
        B[:, :, 0] = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
        C = np.array([0.0, 0.0], dtype=np.float64)
        D = np.array([0.5, 0.5], dtype=np.float64)
        
        A_obj = utils.obj_array(1)
        A_obj[0] = A
        B_obj = utils.obj_array(1)
        B_obj[0] = B
        C_obj = utils.obj_array(1)
        C_obj[0] = C
        D_obj = utils.obj_array(1)
        D_obj[0] = D
        
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
        
        # Test required methods exist
        assert hasattr(agent, 'infer_states')
        assert hasattr(agent, 'infer_policies')
        assert hasattr(agent, 'sample_action')
        
        # Test that methods are callable
        assert callable(agent.infer_states)
        assert callable(agent.infer_policies)
        assert callable(agent.sample_action)
    
    def test_simulation_step_execution(self):
        """Test executing a single simulation step."""
        from pymdp.agent import Agent
        from pymdp import utils
        
        # Create agent
        A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        B = np.zeros((2, 2, 1), dtype=np.float64)
        B[:, :, 0] = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
        C = np.array([0.0, 0.0], dtype=np.float64)
        D = np.array([0.5, 0.5], dtype=np.float64)
        
        A_obj = utils.obj_array(1)
        A_obj[0] = A
        B_obj = utils.obj_array(1)
        B_obj[0] = B
        C_obj = utils.obj_array(1)
        C_obj[0] = C
        D_obj = utils.obj_array(1)
        D_obj[0] = D
        
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
        
        # Execute one step
        obs = np.array([0])  # First observation
        qs = agent.infer_states(obs)
        q_pi, neg_efe = agent.infer_policies()
        action = agent.sample_action()
        
        # Verify results
        assert qs is not None
        assert len(qs) > 0
        assert q_pi is not None
        assert action is not None
        assert len(action) > 0

