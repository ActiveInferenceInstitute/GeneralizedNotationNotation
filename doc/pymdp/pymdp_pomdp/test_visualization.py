#!/usr/bin/env python3
"""
Test script for PyMDP gridworld visualization

This script tests the extracted visualization utilities to ensure
they work correctly without requiring the full PyMDP simulation.
"""

import numpy as np
from pathlib import Path
import tempfile
from pymdp_gridworld_visualizer import GridworldVisualizer, create_visualizer

def test_visualization():
    """Test visualization utilities"""
    
    print("Testing PyMDP gridworld visualization utilities...")
    
    # Create test gridworld layout
    grid_layout = np.array([
        [0, 0, 0, 0, 2],  # Top row: goal at top-right
        [0, 1, 0, 0, 0],  # Wall in middle
        [0, 0, 0, 1, 0],  # Wall in middle
        [0, 0, 0, 0, 0],  # Empty
        [0, 0, 0, 0, 0]   # Bottom row: start area
    ])
    
    print(f"Grid layout shape: {grid_layout.shape}")
    
    # Test basic visualization creation
    print("\n1. Testing visualizer creation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            visualizer = GridworldVisualizer(
                grid_layout=grid_layout,
                output_dir=temp_path,
                show_plots=False  # Don't display plots during testing
            )
            print("✓ Successfully created GridworldVisualizer")
        except Exception as e:
            print(f"✗ Failed to create visualizer: {e}")
            return False
        
        # Test factory function
        try:
            visualizer2 = create_visualizer(grid_layout, temp_path, show_plots=False)
            print("✓ Successfully created visualizer using factory function")
        except Exception as e:
            print(f"✗ Factory function failed: {e}")
            return False
    
    # Test visualization methods (without actual plotting to avoid display issues)
    print("\n2. Testing visualization methods...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = GridworldVisualizer(grid_layout, temp_path, show_plots=False)
        
        # Test basic gridworld plot
        try:
            agent_pos = (2, 2)
            beliefs = np.random.dirichlet(np.ones(25))  # Random belief distribution
            
            fig = visualizer.plot_gridworld(
                agent_pos=agent_pos,
                beliefs=beliefs,
                title="Test Gridworld",
                save_path=temp_path / "test_gridworld.png"
            )
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            # Check if file was created
            if (temp_path / "test_gridworld.png").exists():
                print("✓ Successfully created gridworld plot")
            else:
                print("✗ Gridworld plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Gridworld plotting failed: {e}")
            return False
        
        # Test trajectory plot
        try:
            trajectory = [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
            
            fig = visualizer.plot_trajectory(
                trajectory=trajectory,
                title="Test Trajectory",
                save_path=temp_path / "test_trajectory.png"
            )
            
            plt.close(fig)
            
            if (temp_path / "test_trajectory.png").exists():
                print("✓ Successfully created trajectory plot")
            else:
                print("✗ Trajectory plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Trajectory plotting failed: {e}")
            return False
        
        # Test performance metrics plot
        try:
            test_metrics = {
                'episode_rewards': [9.9, 8.7, 10.0, 9.5, 9.8],
                'episode_lengths': [2, 5, 1, 3, 2],
                'belief_entropies': [2.89, 1.45, 0.73, 1.2, 0.9],
                'success_rates': [1.0, 1.0, 1.0, 1.0, 1.0]
            }
            
            fig = visualizer.plot_performance_metrics(
                metrics=test_metrics,
                save_path=temp_path / "test_performance.png"
            )
            
            plt.close(fig)
            
            if (temp_path / "test_performance.png").exists():
                print("✓ Successfully created performance metrics plot")
            else:
                print("✗ Performance metrics plot file not created")
                return False
                
        except Exception as e:
            print(f"✗ Performance metrics plotting failed: {e}")
            return False
        
        # Test belief evolution plot
        try:
            belief_traces = [
                np.random.dirichlet(np.ones(25)) for _ in range(5)
            ]
            
            fig = visualizer.plot_belief_evolution(
                belief_traces=belief_traces,
                title="Test Belief Evolution",
                save_path=temp_path / "test_belief_evolution.png",
                max_steps=5
            )
            
            if fig:
                plt.close(fig)
                print("✓ Successfully created belief evolution plot")
            else:
                print("✓ Belief evolution plot skipped (empty traces)")
                
        except Exception as e:
            print(f"✗ Belief evolution plotting failed: {e}")
            return False
        
        # Test free energy evolution plot
        try:
            variational_fe = [3.2, 2.8, 2.1, 1.9, 1.5]
            expected_fe = [np.random.random(4) * 5 for _ in range(5)]
            actions = [0, 2, 1, 3, 2]
            
            fig = visualizer.plot_free_energy_evolution(
                variational_fe=variational_fe,
                expected_fe=expected_fe,
                actions=actions,
                title="Test Free Energy Evolution",
                save_path=temp_path / "test_free_energy.png"
            )
            
            if fig:
                plt.close(fig)
                print("✓ Successfully created free energy evolution plot")
            else:
                print("✓ Free energy evolution plot skipped (no data)")
                
        except Exception as e:
            print(f"✗ Free energy evolution plotting failed: {e}")
            return False
        
        # Test Active Inference analysis plot
        try:
            positions = [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
            
            fig = visualizer.plot_active_inference_analysis(
                belief_traces=belief_traces,
                variational_fe=variational_fe,
                expected_fe=expected_fe,
                actions=actions,
                positions=positions,
                episode_num=1,
                save_path=temp_path / "test_active_inference_analysis.png"
            )
            
            if fig:
                plt.close(fig)
                print("✓ Successfully created Active Inference analysis plot")
            else:
                print("✓ Active Inference analysis plot skipped (no data)")
                
        except Exception as e:
            print(f"✗ Active Inference analysis plotting failed: {e}")
            return False
        
        # Test summary figure
        try:
            final_pos = (0, 4)
            final_beliefs = np.random.dirichlet(np.ones(25))
            
            fig = visualizer.create_summary_figure(
                final_pos=final_pos,
                final_beliefs=final_beliefs,
                trajectory=trajectory,
                metrics=test_metrics,
                episode_num=1,
                save_path=temp_path / "test_summary.png"
            )
            
            plt.close(fig)
            
            if (temp_path / "test_summary.png").exists():
                print("✓ Successfully created summary figure")
            else:
                print("✗ Summary figure file not created")
                return False
                
        except Exception as e:
            print(f"✗ Summary figure creation failed: {e}")
            return False
        
        # Check all created files
        png_files = list(temp_path.glob("*.png"))
        print(f"\n3. Verification: Created {len(png_files)} visualization files")
        for png_file in png_files:
            print(f"   - {png_file.name} ({png_file.stat().st_size} bytes)")
    
    print("\n🎉 All visualization tests passed!")
    return True

if __name__ == "__main__":
    success = test_visualization()
    exit(0 if success else 1) 