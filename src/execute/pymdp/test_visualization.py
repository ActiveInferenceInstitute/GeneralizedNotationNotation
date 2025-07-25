#!/usr/bin/env python3
"""
Test PyMDP Visualization Module

Test script for the PyMDP visualization capabilities within the GNN pipeline.
This verifies that visualization works with pipeline-configured simulations.

Features:
- Test discrete state visualization
- Test belief distribution plots 
- Test performance metrics
- Pipeline integration verification

Author: GNN PyMDP Integration
Date: 2024
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from execute.pymdp.pymdp_visualizer import PyMDPVisualizer
from execute.pymdp.pymdp_utils import create_output_directory_with_timestamp

def create_test_data():
    """Create test simulation data for visualization testing."""
    
    # Simple discrete POMDP test data
    num_timesteps = 50
    num_states = 9  # 3x3 grid
    num_observations = 9  # Full observability for testing
    num_actions = 4  # Up, Down, Left, Right
    
    # Generate synthetic trajectory data
    states = np.random.randint(0, num_states, num_timesteps)
    observations = states  # Full observability for testing
    actions = np.random.randint(0, num_actions, num_timesteps)
    rewards = np.random.randn(num_timesteps)
    
    # Generate synthetic belief distributions
    beliefs = []
    for t in range(num_timesteps):
        belief = np.random.dirichlet(np.ones(num_states))
        beliefs.append(belief)
    beliefs = np.array(beliefs)
    
    # Generate synthetic agent trace
    agent_trace = {
        'posterior_states': [beliefs[t] for t in range(num_timesteps)],
        'posterior_policies': [[0.7, 0.2, 0.1] for _ in range(num_timesteps)],
        'free_energy': np.random.randn(num_timesteps),
        'expected_free_energy': np.random.randn(num_timesteps),
        'policy_precision': np.ones(num_timesteps) * 2.0
    }
    
    # Model configuration
    model_config = {
        'num_states': num_states,
        'num_observations': num_observations, 
        'num_actions': num_actions,
        'grid_size': 3,  # 3x3 grid
        'goal_location': 8,  # Bottom-right corner
        'agent_start': 0   # Top-left corner
    }
    
    return {
        'states': states,
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'beliefs': beliefs,
        'agent_trace': agent_trace,
        'model_config': model_config
    }

def test_visualizer():
    """Test the PyMDPVisualizer with synthetic data."""
    
    print("Testing PyMDP Visualization Module...")
    
    # Create test output directory
    output_dir = create_output_directory_with_timestamp("test_pymdp_viz")
    print(f"Output directory: {output_dir}")
    
    # Create test data
    test_data = create_test_data()
    
    # Initialize visualizer
    visualizer = PyMDPVisualizer(
        output_dir=output_dir,
        grid_size=test_data['model_config']['grid_size']
    )
    
    # Test 1: Create trajectory visualization
    print("Creating trajectory visualization...")
    try:
        visualizer.plot_trajectory(
            states=test_data['states'],
            actions=test_data['actions'],
            grid_size=test_data['model_config']['grid_size'],
            goal_location=test_data['model_config']['goal_location'],
            save_path=output_dir / "trajectory.png"
        )
        print("✓ Trajectory visualization created successfully")
    except Exception as e:
        print(f"✗ Trajectory visualization failed: {e}")
    
    # Test 2: Create belief distribution plots
    print("Creating belief distribution plots...")
    try:
        visualizer.plot_belief_evolution(
            beliefs=test_data['beliefs'],
            save_path=output_dir / "beliefs.png"
        )
        print("✓ Belief distribution plots created successfully")
    except Exception as e:
        print(f"✗ Belief distribution plots failed: {e}")
    
    # Test 3: Create performance metrics
    print("Creating performance metrics...")
    try:
        visualizer.plot_performance_metrics(
            rewards=test_data['rewards'],
            free_energy=test_data['agent_trace']['free_energy'],
            save_path=output_dir / "performance.png"
        )
        print("✓ Performance metrics created successfully")
    except Exception as e:
        print(f"✗ Performance metrics failed: {e}")
    
    # Test 4: Create all visualizations
    print("Creating all visualizations...")
    try:
        visualizer.create_all_visualizations(
            simulation_data=test_data,
            title_prefix="Test"
        )
        print("✓ All visualizations created successfully")
    except Exception as e:
        print(f"✗ All visualizations failed: {e}")
    
    print(f"\nVisualization test completed. Check output directory: {output_dir}")
    return output_dir

def test_pipeline_integration():
    """Test integration with pipeline configuration."""
    
    print("\nTesting pipeline integration...")
    
    # Test GNN-style configuration
    gnn_config = {
        'model_name': 'test_pomdp',
        'num_states': 9,
        'num_observations': 9,
        'num_actions': 4,
        'grid_size': 3,
        'goal_state': 8,
        'start_state': 0,
        'num_timesteps': 50
    }
    
    # Create test data with GNN configuration
    test_data = create_test_data()
    test_data['gnn_config'] = gnn_config
    
    output_dir = create_output_directory_with_timestamp("test_pipeline_integration")
    
    visualizer = PyMDPVisualizer(output_dir=output_dir)
    
    try:
        # Test that visualizer can handle GNN-style configuration
        visualizer.create_all_visualizations(
            simulation_data=test_data,
            title_prefix="Pipeline Test"
        )
        print("✓ Pipeline integration test successful")
        return True
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    output_dir = test_visualizer()
    integration_success = test_pipeline_integration()
    
    print(f"\nTest Summary:")
    print(f"Visualization test output: {output_dir}")
    print(f"Pipeline integration: {'✓ PASSED' if integration_success else '✗ FAILED'}")
    
    print(f"\nTo view results:")
    print(f"ls -la {output_dir}")
    print(f"open {output_dir}") 