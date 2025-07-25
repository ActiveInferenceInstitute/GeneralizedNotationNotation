#!/usr/bin/env python3
"""
Test script for PyMDP visualizer

This script tests the PyMDP visualization utilities to ensure
they work correctly for discrete POMDP simulations.
"""

import numpy as np
from pathlib import Path
import tempfile
from .pymdp_visualizer import PyMDPVisualizer, create_visualizer

def test_visualizer_creation():
    """Test PyMDP visualizer creation."""
    
    print("Testing PyMDP visualizer creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            visualizer = PyMDPVisualizer(
                output_dir=temp_path,
                show_plots=False  # Don't display plots during testing
            )
            print("✓ Successfully created PyMDPVisualizer")
        except Exception as e:
            print(f"✗ Failed to create visualizer: {e}")
            return False
        
        # Test factory function
        try:
            visualizer2 = create_visualizer(temp_path, show_plots=False)
            print("✓ Successfully created visualizer using factory function")
        except Exception as e:
            print(f"✗ Factory function failed: {e}")
            return False
    
    return True

def test_discrete_state_visualization():
    """Test discrete state visualization."""
    
    print("\nTesting discrete state visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            # Test discrete state sequence
            state_sequence = [0, 1, 2, 1, 0, 2]
            num_states = 3
            
            fig = visualizer.plot_discrete_states(
                state_sequence=state_sequence,
                num_states=num_states,
                title="Test State Sequence",
                save_path=temp_path / "test_states.png"
            )
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            # Check if file was created
            if (temp_path / "test_states.png").exists():
                print("✓ Successfully created discrete state plot")
            else:
                print("✗ State plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Discrete state plotting failed: {e}")
            return False
    
    return True

def test_belief_visualization():
    """Test belief evolution visualization."""
    
    print("\nTesting belief evolution visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            # Create test belief traces
            belief_traces = [
                np.array([0.8, 0.2, 0.0]),
                np.array([0.6, 0.3, 0.1]),
                np.array([0.2, 0.3, 0.5]),
                np.array([0.1, 0.1, 0.8])
            ]
            
            fig = visualizer.plot_belief_evolution(
                belief_traces=belief_traces,
                title="Test Belief Evolution",
                save_path=temp_path / "test_beliefs.png"
            )
            
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
                print("✓ Successfully created belief evolution plot")
            else:
                print("✓ Belief evolution plot skipped (empty traces)")
                
        except Exception as e:
            print(f"✗ Belief evolution plotting failed: {e}")
            return False
    
    return True

def test_performance_metrics():
    """Test performance metrics visualization."""
    
    print("\nTesting performance metrics visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            test_metrics = {
                'episode_rewards': [0.9, 1.2, 0.8, 1.5, 1.0],
                'episode_lengths': [15, 12, 18, 10, 14],
                'belief_entropies': [1.8, 1.2, 0.9, 1.1, 1.0],
                'success_rates': [0.2, 0.4, 0.6, 0.75, 0.8]
            }
            
            fig = visualizer.plot_performance_metrics(
                metrics=test_metrics,
                save_path=temp_path / "test_performance.png"
            )
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            if (temp_path / "test_performance.png").exists():
                print("✓ Successfully created performance metrics plot")
            else:
                print("✗ Performance metrics plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Performance metrics plotting failed: {e}")
            return False
    
    return True

def test_action_visualization():
    """Test action sequence visualization."""
    
    print("\nTesting action sequence visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            action_sequence = [0, 1, 2, 0, 1, 2, 1, 0]
            num_actions = 3
            
            fig = visualizer.plot_action_sequence(
                action_sequence=action_sequence,
                num_actions=num_actions,
                title="Test Actions",
                save_path=temp_path / "test_actions.png"
            )
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            if (temp_path / "test_actions.png").exists():
                print("✓ Successfully created action sequence plot")
            else:
                print("✗ Action sequence plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Action sequence plotting failed: {e}")
            return False
    
    return True

def test_observation_visualization():
    """Test observation sequence visualization."""
    
    print("\nTesting observation sequence visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            observation_sequence = [0, 0, 1, 2, 1, 0, 2]
            num_observations = 3
            
            fig = visualizer.plot_observation_sequence(
                observation_sequence=observation_sequence,
                num_observations=num_observations,
                title="Test Observations",
                save_path=temp_path / "test_observations.png"
            )
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            if (temp_path / "test_observations.png").exists():
                print("✓ Successfully created observation sequence plot")
            else:
                print("✗ Observation sequence plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Observation sequence plotting failed: {e}")
            return False
    
    return True

def test_episode_summary():
    """Test episode summary visualization."""
    
    print("\nTesting episode summary visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            episode_trace = {
                'true_states': [0, 1, 2, 1, 0],
                'observations': [0, 1, 2, 1, 0],
                'actions': [1, 2, 0, 1],
                'rewards': [-0.1, -0.1, 1.0, -0.1],
                'beliefs': [
                    np.array([0.8, 0.2, 0.0]),
                    np.array([0.3, 0.6, 0.1]),
                    np.array([0.1, 0.2, 0.7]),
                    np.array([0.2, 0.7, 0.1]),
                    np.array([0.8, 0.1, 0.1])
                ]
            }
            
            fig = visualizer.plot_episode_summary(
                episode_trace=episode_trace,
                episode_num=1,
                save_path=temp_path / "test_episode_summary.png"
            )
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            if (temp_path / "test_episode_summary.png").exists():
                print("✓ Successfully created episode summary")
            else:
                print("✗ Episode summary file not created")
                return False
                
        except Exception as e:
            print(f"✗ Episode summary creation failed: {e}")
            return False
    
    return True

def test_comprehensive_visualization():
    """Test comprehensive visualization generation."""
    
    print("\nTesting comprehensive visualization generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)
        
        try:
            # Create test data
            all_traces = [
                {
                    'true_states': [0, 1, 2],
                    'observations': [0, 1, 2],
                    'actions': [1, 2],
                    'rewards': [-0.1, 1.0],
                    'beliefs': [
                        np.array([0.8, 0.2, 0.0]),
                        np.array([0.3, 0.6, 0.1]),
                        np.array([0.1, 0.2, 0.7])
                    ]
                },
                {
                    'true_states': [2, 1, 0],
                    'observations': [2, 1, 0],
                    'actions': [0, 1],
                    'rewards': [-0.1, -0.1],
                    'beliefs': [
                        np.array([0.1, 0.2, 0.7]),
                        np.array([0.2, 0.7, 0.1]),
                        np.array([0.8, 0.1, 0.1])
                    ]
                }
            ]
            
            performance_metrics = {
                'episode_rewards': [0.9, -0.2],
                'episode_lengths': [3, 3],
                'belief_entropies': [1.2, 1.0],
                'success_rates': [0.5, 0.5]
            }
            
            # Generate comprehensive visualizations
            visualizer.create_comprehensive_visualizations(all_traces, performance_metrics)
            
            # Check if files were created
            png_files = list(temp_path.glob("**/*.png"))
            print(f"✓ Created {len(png_files)} visualization files")
            
            if len(png_files) > 0:
                print("✓ Comprehensive visualization generation successful")
                for png_file in png_files[:3]:  # Show first 3 as examples
                    print(f"   - {png_file.name} ({png_file.stat().st_size} bytes)")
            else:
                print("✗ No visualization files created")
                return False
                
        except Exception as e:
            print(f"✗ Comprehensive visualization failed: {e}")
            return False
    
    return True

def main():
    """Run all visualization tests."""
    print("=" * 60)
    print("PYMDP VISUALIZER TEST SUITE")
    print("=" * 60)
    
    success = True
    
    success &= test_visualizer_creation()
    success &= test_discrete_state_visualization()
    success &= test_belief_visualization()
    success &= test_performance_metrics()
    success &= test_action_visualization()
    success &= test_observation_visualization()
    success &= test_episode_summary()
    success &= test_comprehensive_visualization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL VISUALIZATION TESTS PASSED!")
    else:
        print("❌ SOME VISUALIZATION TESTS FAILED!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 