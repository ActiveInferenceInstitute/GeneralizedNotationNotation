import pytest
import json
import numpy as np
from pathlib import Path
from analysis.processor import process_analysis

class TestAnalysisOverall:
    """Test suite for Analysis module."""

    @pytest.fixture
    def sample_gnn_for_analysis(self, safe_filesystem):
        """Create a sample GNN file to analyze."""
        content = """
# Analysis Target

## StateSpaceBlock
s[10, type=float]

## Connections
s->s

## Time
Dynamic
"""
        return safe_filesystem.create_file("model_analysis.md", content)

    def test_process_analysis_flow(self, safe_filesystem, sample_gnn_for_analysis):
        """Test the analysis processing workflow."""
        target_dir = sample_gnn_for_analysis.parent
        output_dir = safe_filesystem.create_dir("analysis_output")
        
        # Need to ensure submodules of analysis don't crash.
        # analysis/processor imports from analyzer.py.
        # analyzer.py imports numpy etc.
        
        try:
            success = process_analysis(target_dir, output_dir, verbose=True)
            assert success is True
            
            # Check results
            results_dir = output_dir / "analysis_results"
            assert results_dir.exists()
            assert (results_dir / "analysis_results.json").exists()
            assert (results_dir / "analysis_summary.md").exists()
            
            with open(results_dir / "analysis_results.json", 'r') as f:
                data = json.load(f)
            
            assert data["processed_files"] == 1
            assert len(data["statistical_analysis"]) == 1
            
        except ImportError:
            pytest.skip("Skipping analysis test due to missing dependencies (numpy/matplotlib)")
        except Exception as e:
            pytest.fail(f"Analysis processing failed: {e}")

    def test_process_analysis_no_files(self, safe_filesystem):
        """Test behavior with no files."""
        empty_dir = safe_filesystem.create_dir("empty")
        output_dir = safe_filesystem.create_dir("output")
        
        success = process_analysis(empty_dir, output_dir)
        
        # Processor returns False if no files found (based on source: results["success"] = False)
        assert success is False


class TestPostSimulationVisualization:
    """Test suite for post-simulation visualization functions."""

    def test_generate_belief_heatmaps(self, safe_filesystem):
        """Test belief heatmap generation."""
        try:
            from analysis.post_simulation import generate_belief_heatmaps
            
            # Create sample beliefs data: 10 timesteps, 3 states
            beliefs = [
                [0.8, 0.1, 0.1],
                [0.7, 0.2, 0.1],
                [0.5, 0.3, 0.2],
                [0.3, 0.5, 0.2],
                [0.2, 0.6, 0.2],
                [0.1, 0.7, 0.2],
                [0.1, 0.5, 0.4],
                [0.1, 0.3, 0.6],
                [0.1, 0.2, 0.7],
                [0.1, 0.1, 0.8],
            ]
            
            output_dir = safe_filesystem.create_dir("viz_output")
            output_file = output_dir / "belief_heatmap.png"
            
            result = generate_belief_heatmaps(beliefs, output_file, "Test Belief Heatmap")
            
            assert result == str(output_file)
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
        except ImportError:
            pytest.skip("Missing matplotlib for visualization tests")

    def test_generate_action_analysis(self, safe_filesystem):
        """Test action analysis visualization."""
        try:
            from analysis.post_simulation import generate_action_analysis
            
            # Create sample actions: 20 timesteps with 3 possible actions
            actions = [0, 1, 2, 0, 1, 1, 2, 2, 0, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2]
            
            output_dir = safe_filesystem.create_dir("viz_output")
            output_file = output_dir / "action_analysis.png"
            
            result = generate_action_analysis(actions, output_file, "Test Action Analysis")
            
            assert result == str(output_file)
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
        except ImportError:
            pytest.skip("Missing matplotlib for visualization tests")

    def test_generate_free_energy_plots(self, safe_filesystem):
        """Test free energy plot generation."""
        try:
            from analysis.post_simulation import generate_free_energy_plots
            
            # Create sample free energy values: 20 timesteps, generally decreasing
            free_energy = [10.0, 9.5, 9.0, 8.5, 8.2, 7.8, 7.5, 7.2, 7.0, 6.8,
                          6.5, 6.3, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.5, 5.5]
            
            output_dir = safe_filesystem.create_dir("viz_output")
            output_file = output_dir / "free_energy.png"
            
            result = generate_free_energy_plots(free_energy, output_file, "Test Free Energy")
            
            assert result == str(output_file)
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
        except ImportError:
            pytest.skip("Missing matplotlib for visualization tests")

    def test_generate_observation_analysis(self, safe_filesystem):
        """Test observation analysis visualization."""
        try:
            from analysis.post_simulation import generate_observation_analysis
            
            # Create sample observations: 15 timesteps with 3 possible observations
            observations = [0, 1, 0, 2, 1, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1]
            
            output_dir = safe_filesystem.create_dir("viz_output")
            output_file = output_dir / "observations.png"
            
            result = generate_observation_analysis(observations, output_file, "Test Observations")
            
            assert result == str(output_file)
            assert output_file.exists()
            assert output_file.stat().st_size > 0
            
        except ImportError:
            pytest.skip("Missing matplotlib for visualization tests")

    def test_analyze_free_energy(self):
        """Test free energy analysis function."""
        from analysis.post_simulation import analyze_free_energy
        
        # Test decreasing free energy (good Active Inference behavior)
        fe_values = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.2, 2.1, 2.05, 2.02]
        
        result = analyze_free_energy(fe_values, "pymdp", "test_model")
        
        assert result["framework"] == "pymdp"
        assert result["model_name"] == "test_model"
        assert result["free_energy_count"] == 10
        assert "mean_free_energy" in result
        assert "std_free_energy" in result
        assert result["free_energy_decreasing"] == True  # Trend should be negative

    def test_analyze_simulation_traces(self):
        """Test simulation trace analysis function."""
        from analysis.post_simulation import analyze_simulation_traces
        
        # Test with list-based traces
        traces = [
            [0, 1, 2, 1, 0],  # Trace 1
            [1, 2, 2, 0, 1, 2],  # Trace 2
            [0, 0, 1, 2],  # Trace 3
        ]
        
        result = analyze_simulation_traces(traces, "rxinfer", "test_model")
        
        assert result["framework"] == "rxinfer"
        assert result["trace_count"] == 3
        assert result["trace_lengths"] == [5, 6, 4]
        assert result["avg_trace_length"] == 5.0

    def test_analyze_policy_convergence(self):
        """Test policy convergence analysis."""
        from analysis.post_simulation import analyze_policy_convergence
        
        # Test with converging policies
        policy_traces = [
            [0.33, 0.33, 0.34],  # Uniform at start
            [0.4, 0.3, 0.3],
            [0.5, 0.25, 0.25],
            [0.7, 0.15, 0.15],
            [0.9, 0.05, 0.05],  # Converged to action 0
        ]
        
        result = analyze_policy_convergence(policy_traces, "jax", "test_model")
        
        assert result["framework"] == "jax"
        assert result["policy_count"] == 5
        assert len(result["policy_entropy"]) == 5
        # First entropy should be higher than last (convergence)
        assert result["policy_entropy"][0] > result["policy_entropy"][-1]

    def test_compare_framework_results(self):
        """Test cross-framework comparison."""
        from analysis.post_simulation import compare_framework_results
        
        framework_results = {
            "pymdp": {
                "success": True,
                "execution_time": 1.5,
                "simulation_data": {"free_energy": [10.0, 8.0, 6.0]}
            },
            "rxinfer": {
                "success": True,
                "execution_time": 0.8,
                "simulation_data": {"free_energy": [12.0, 9.0, 7.0]}
            }
        }
        
        result = compare_framework_results(framework_results, "test_model")
        
        assert result["framework_count"] == 2
        assert "pymdp" in result["frameworks_compared"]
        assert "rxinfer" in result["frameworks_compared"]
        # RxInfer should be faster
        assert result["comparisons"]["fastest_execution"]["framework"] == "rxinfer"


class TestAnalysisModuleImports:
    """Test that all new visualization functions are properly exported."""

    def test_visualization_function_exports(self):
        """Test that new visualization functions are exported from analysis module."""
        from analysis import (
            visualize_all_framework_outputs,
            generate_belief_heatmaps,
            generate_action_analysis,
            generate_free_energy_plots,
            generate_observation_analysis,
            generate_cross_framework_comparison,
            plot_belief_evolution,
            animate_belief_evolution
        )
        
        # All functions should be callable
        assert callable(visualize_all_framework_outputs)
        assert callable(generate_belief_heatmaps)
        assert callable(generate_action_analysis)
        assert callable(generate_free_energy_plots)
        assert callable(generate_observation_analysis)
        assert callable(generate_cross_framework_comparison)
        assert callable(plot_belief_evolution)
        assert callable(animate_belief_evolution)
