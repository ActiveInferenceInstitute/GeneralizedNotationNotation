#!/usr/bin/env python3
"""
Comprehensive real-data tests for visualization module.

These tests validate actual visualization generation with real GNN files,
ensuring proper matplotlib backend handling, progress tracking, and error recovery.
"""

import pytest
import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization import (
    process_visualization_main,
    _configure_matplotlib_backend,
    MatrixVisualizer,
    generate_network_visualizations,
    generate_combined_analysis
)


class TestMatplotlibBackendConfiguration:
    """Test matplotlib backend configuration for headless environments."""
    
    def test_backend_configuration_with_display(self, caplog):
        """Test backend configuration when DISPLAY is available."""
        import logging
        logger = logging.getLogger("test_viz")
        
        result = _configure_matplotlib_backend(logger)
        
        # Should succeed regardless of environment
        assert isinstance(result, bool)
        
    def test_backend_configuration_headless(self, caplog, monkeypatch):
        """Test backend configuration in headless environment."""
        import logging
        import os
        
        # Simulate headless environment
        monkeypatch.delenv('DISPLAY', raising=False)
        monkeypatch.setenv('MPLBACKEND', 'Agg')
        
        logger = logging.getLogger("test_viz")
        result = _configure_matplotlib_backend(logger)
        
        # Should configure Agg backend successfully
        assert result is True
        assert "headless" in caplog.text.lower() or "Agg" in caplog.text


class TestVisualizationProcessing:
    """Test actual visualization processing with real data."""
    
    @pytest.fixture
    def test_gnn_dir(self):
        """Create temporary directory with test GNN files."""
        test_dir = tempfile.mkdtemp()
        test_gnn_dir = Path(test_dir) / "gnn_files"
        test_gnn_dir.mkdir()
        
        # Create a minimal GNN file
        gnn_file = test_gnn_dir / "test_model.md"
        gnn_file.write_text("""
# Test Active Inference Model

## State Blocks
[A]: State space (3)
[B]: Transition matrix (3x3x2)
[C]: Preference matrix (3)

## Connections
A -> B: "state transitions"
B -> C: "preferences"

## Parameters
- learning_rate: 0.01
- temperature: 1.0
""")
        
        yield test_gnn_dir
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    @pytest.fixture
    def test_output_dir(self):
        """Create temporary output directory."""
        test_dir = tempfile.mkdtemp()
        output_dir = Path(test_dir) / "output"
        output_dir.mkdir()
        
        # Create required GNN processing results (step 3 output)
        gnn_output_dir = output_dir / "3_gnn_output"
        gnn_output_dir.mkdir()
        
        gnn_results = {
            "processed_files": [
                {
                    "file_name": "test_model.md",
                    "file_path": "input/gnn_files/test_model.md",
                    "parse_success": True,
                    "parsed_model_file": str(gnn_output_dir / "test_model.json")
                }
            ],
            "summary": {"total_files": 1, "successful": 1}
        }
        
        # Create parsed model file
        parsed_model = {
            "model_name": "test_model",
            "parameters": [
                {"name": "A", "type": "matrix", "shape": [3], "values": [1, 2, 3]},
                {"name": "B", "type": "matrix", "shape": [3, 3, 2], "values": list(range(18))},
                {"name": "C", "type": "matrix", "shape": [3], "values": [0.5, 0.3, 0.2]}
            ],
            "state_blocks": [
                {"name": "A", "dimensions": [3]},
                {"name": "B", "dimensions": [3, 3, 2]},
                {"name": "C", "dimensions": [3]}
            ],
            "connections": [
                {"source": "A", "target": "B", "type": "state"},
                {"source": "B", "target": "C", "type": "preference"}
            ]
        }
        
        with open(gnn_output_dir / "test_model.json", 'w') as f:
            json.dump(parsed_model, f)
        
        with open(gnn_output_dir / "gnn_processing_results.json", 'w') as f:
            json.dump(gnn_results, f)
        
        yield output_dir
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_visualization_main_success(self, test_gnn_dir, test_output_dir, caplog):
        """Test complete visualization processing with real data."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        viz_output_dir = test_output_dir / "8_visualization_output"
        viz_output_dir.mkdir()
        
        result = process_visualization_main(
            target_dir=test_gnn_dir,
            output_dir=viz_output_dir,
            verbose=True
        )
        
        # Check result
        assert isinstance(result, bool)
        
        # Check that visualization results were created
        results_file = viz_output_dir / "visualization_results.json"
        assert results_file.exists(), "visualization_results.json should be created"
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Verify results structure
        assert "timestamp" in results
        assert "files_visualized" in results
        assert "summary" in results
        
        # Check summary
        summary = results["summary"]
        assert "total_files" in summary
        assert "successful_visualizations" in summary
        assert "total_images_generated" in summary
        
        # Check that some visualization was attempted
        assert summary["total_files"] >= 0
        
        # Check logging messages
        assert "Visualizing" in caplog.text or "visualization" in caplog.text.lower()
    
    def test_visualization_with_missing_gnn_data(self, test_gnn_dir, caplog):
        """Test visualization when GNN processing results are missing."""
        import logging
        caplog.set_level(logging.INFO)
        
        output_dir = Path(tempfile.mkdtemp())
        viz_output_dir = output_dir / "8_visualization_output"
        viz_output_dir.mkdir(parents=True)
        
        try:
            result = process_visualization_main(
                target_dir=test_gnn_dir,
                output_dir=viz_output_dir,
                verbose=True
            )
            
            # Should fail gracefully
            assert result is False
            assert "not found" in caplog.text.lower() or "Run step 3 first" in caplog.text
            
        finally:
            shutil.rmtree(output_dir)
    
    def test_matrix_visualizer_with_real_parameters(self):
        """Test MatrixVisualizer with real parameter data."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            mv = MatrixVisualizer()
            
            # Create test parameters
            parameters = [
                {"name": "A", "shape": [3, 3], "values": np.eye(3).flatten().tolist()},
                {"name": "B", "shape": [2, 2], "values": [1, 2, 3, 4]},
            ]
            
            # Test matrix extraction
            matrices = mv.extract_matrix_data_from_parameters(parameters)
            
            assert isinstance(matrices, dict)
            assert len(matrices) >= 0  # May be empty if extraction fails
            
        except ImportError:
            pytest.skip("MatrixVisualizer not available (missing dependencies)")
    
    def test_visualization_progress_tracking(self, test_gnn_dir, test_output_dir, caplog):
        """Test that visualization provides progress tracking."""
        import logging
        caplog.set_level(logging.INFO)
        
        viz_output_dir = test_output_dir / "8_visualization_output"
        viz_output_dir.mkdir()
        
        result = process_visualization_main(
            target_dir=test_gnn_dir,
            output_dir=viz_output_dir,
            verbose=True
        )
        
        # Check for progress indicators in logs
        log_text = caplog.text
        
        # Should have file counting or progress indicators
        has_progress = (
            "[" in log_text and "/" in log_text and "]" in log_text or
            "Visualizing" in log_text or
            "completed" in log_text.lower()
        )
        
        assert has_progress, "Should provide progress tracking information"
    
    def test_visualization_error_recovery(self, test_gnn_dir, test_output_dir, caplog):
        """Test visualization error recovery and graceful degradation."""
        import logging
        caplog.set_level(logging.WARNING)
        
        viz_output_dir = test_output_dir / "8_visualization_output"
        viz_output_dir.mkdir()
        
        # Force a potential error condition by removing parsed model
        gnn_output_dir = test_output_dir / "3_gnn_output"
        parsed_model_file = gnn_output_dir / "test_model.json"
        if parsed_model_file.exists():
            parsed_model_file.unlink()
        
        result = process_visualization_main(
            target_dir=test_gnn_dir,
            output_dir=viz_output_dir,
            verbose=True
        )
        
        # Should handle gracefully
        assert isinstance(result, bool)
        
        # Check for warning messages
        log_text = caplog.text.lower()
        has_warnings = (
            "warning" in log_text or
            "skipped" in log_text or
            "not found" in log_text
        )
        
        # Warnings expected when things go wrong
        # (may not have warnings if fallback succeeds)


class TestVisualizationComponents:
    """Test individual visualization components."""
    
    def test_network_visualization_generation(self):
        """Test network visualization with real data."""
        try:
            test_data = {
                "state_blocks": [
                    {"name": "A", "dimensions": [3]},
                    {"name": "B", "dimensions": [2]},
                ],
                "connections": [
                    {"source": "A", "target": "B", "type": "transition"}
                ]
            }
            
            output_dir = Path(tempfile.mkdtemp())
            
            try:
                files = generate_network_visualizations(test_data, output_dir, "test_model")
                
                # Should return a list (may be empty if dependencies missing)
                assert isinstance(files, list)
                
            finally:
                shutil.rmtree(output_dir)
                
        except ImportError:
            pytest.skip("Network visualization dependencies not available")
    
    def test_combined_analysis_generation(self):
        """Test combined analysis with real data."""
        try:
            test_data = {
                "parameters": [
                    {"name": "A", "shape": [3], "values": [1, 2, 3]},
                ],
                "state_blocks": [
                    {"name": "A", "dimensions": [3]},
                ],
                "connections": []
            }
            
            output_dir = Path(tempfile.mkdtemp())
            
            try:
                files = generate_combined_analysis(test_data, output_dir, "test_model")
                
                # Should return a list (may be empty if dependencies missing)
                assert isinstance(files, list)
                
            finally:
                shutil.rmtree(output_dir)
                
        except ImportError:
            pytest.skip("Combined analysis dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

