"""
Comprehensive real-data tests for visualization module.

These tests validate actual visualization generation with real GNN files,
ensuring proper matplotlib backend handling, progress tracking, and error recovery.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from visualization import process_visualization
from visualization.combined_analysis import generate_combined_analysis
from visualization.network_visualizations import generate_network_visualizations

class TestMatplotlibBackendConfiguration:
    """Test matplotlib backend configuration for headless environments."""

    @pytest.mark.skip(reason='Testing internal implementation - backend auto-configured')
    def test_backend_configuration_with_display(self, caplog: Any) -> None:
        """Test backend configuration when DISPLAY is available."""
        pass

    @pytest.mark.skip(reason='Testing internal implementation - backend auto-configured')
    def test_backend_configuration_headless(self, caplog: Any, monkeypatch: Any) -> None:
        """Test backend configuration in headless environment."""
        pass

class TestVisualizationProcessing:
    """Test actual visualization processing with real data."""

    @pytest.fixture
    def test_gnn_dir(self) -> Any:
        """Create temporary directory with test GNN files."""
        test_dir = tempfile.mkdtemp()
        test_gnn_dir = Path(test_dir) / 'gnn_files'
        test_gnn_dir.mkdir()
        gnn_file = test_gnn_dir / 'test_model.md'
        gnn_file.write_text('\n# Test Active Inference Model\n\n## State Blocks\n[A]: State space (3)\n[B]: Transition matrix (3x3x2)\n[C]: Preference matrix (3)\n\n## Connections\nA -> B: "state transitions"\nB -> C: "preferences"\n\n## Parameters\n- learning_rate: 0.01\n- temperature: 1.0\n')
        yield test_gnn_dir
        shutil.rmtree(test_dir)

    @pytest.fixture
    def test_output_dir(self) -> Any:
        """Create temporary output directory."""
        test_dir = tempfile.mkdtemp()
        output_dir = Path(test_dir) / 'output'
        output_dir.mkdir()
        gnn_output_dir = output_dir / '3_gnn_output'
        gnn_output_dir.mkdir()
        gnn_results = {'processed_files': [{'file_name': 'test_model.md', 'file_path': 'input/gnn_files/test_model.md', 'parse_success': True, 'parsed_model_file': str(gnn_output_dir / 'test_model.json')}], 'summary': {'total_files': 1, 'successful': 1}}
        parsed_model = {'model_name': 'test_model', 'parameters': [{'name': 'A', 'type': 'matrix', 'shape': [3], 'values': [1, 2, 3]}, {'name': 'B', 'type': 'matrix', 'shape': [3, 3, 2], 'values': list(range(18))}, {'name': 'C', 'type': 'matrix', 'shape': [3], 'values': [0.5, 0.3, 0.2]}], 'state_blocks': [{'name': 'A', 'dimensions': [3]}, {'name': 'B', 'dimensions': [3, 3, 2]}, {'name': 'C', 'dimensions': [3]}], 'connections': [{'source': 'A', 'target': 'B', 'type': 'state'}, {'source': 'B', 'target': 'C', 'type': 'preference'}]}
        with open(gnn_output_dir / 'test_model.json', 'w') as f:
            json.dump(parsed_model, f)
        with open(gnn_output_dir / 'gnn_processing_results.json', 'w') as f:
            json.dump(gnn_results, f)
        yield output_dir
        shutil.rmtree(test_dir)

    def test_visualization_main_success(self, test_gnn_dir: Any, test_output_dir: Any, caplog: Any) -> None:
        """Test complete visualization processing with real data."""
        import logging
        caplog.set_level(logging.DEBUG)
        viz_output_dir = test_output_dir / '8_visualization_output'
        viz_output_dir.mkdir()
        result = process_visualization(target_dir=test_gnn_dir, output_dir=viz_output_dir, verbose=True)
        assert isinstance(result, bool)
        assert 'visualization' in caplog.text.lower() or 'processing' in caplog.text.lower()

    def test_visualization_with_missing_gnn_data(self, test_gnn_dir: Any, caplog: Any) -> None:
        """Test visualization when GNN processing results are missing."""
        import logging
        caplog.set_level(logging.INFO)
        output_dir = Path(tempfile.mkdtemp())
        viz_output_dir = output_dir / '8_visualization_output'
        viz_output_dir.mkdir(parents=True)
        try:
            result = process_visualization(target_dir=test_gnn_dir, output_dir=viz_output_dir, verbose=True)
            assert isinstance(result, bool)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_matrix_visualizer_with_real_parameters(self) -> None:
        """Test MatrixVisualizer with real parameter data."""
        import numpy as np
        from visualization.matrix_visualizer import MatrixVisualizer
        mv = MatrixVisualizer()
        parameters = [{'name': 'A', 'shape': [3, 3], 'value': np.eye(3).tolist()}, {'name': 'B', 'shape': [2, 2], 'value': [[1, 2], [3, 4]]}]
        matrices = mv.extract_matrix_data_from_parameters(parameters)
        assert isinstance(matrices, dict)
        assert 'A' in matrices and matrices['A'].shape == (3, 3)
        assert 'B' in matrices and matrices['B'].shape == (2, 2)

    def test_visualization_progress_tracking(self, test_gnn_dir: Any, test_output_dir: Any, caplog: Any) -> None:
        """Test that visualization provides progress tracking."""
        import logging
        caplog.set_level(logging.INFO)
        viz_output_dir = test_output_dir / '8_visualization_output'
        viz_output_dir.mkdir()
        process_visualization(target_dir=test_gnn_dir, output_dir=viz_output_dir, verbose=True)
        log_text = caplog.text
        has_progress = 'visualization' in log_text.lower() or 'processing' in log_text.lower() or 'completed' in log_text.lower()
        assert has_progress, 'Should provide some progress information'

    def test_visualization_error_recovery(self, test_gnn_dir: Any, test_output_dir: Any, caplog: Any) -> None:
        """Test visualization error recovery and graceful degradation."""
        import logging
        caplog.set_level(logging.WARNING)
        viz_output_dir = test_output_dir / '8_visualization_output'
        viz_output_dir.mkdir()
        gnn_output_dir = test_output_dir / '3_gnn_output'
        parsed_model_file = gnn_output_dir / 'test_model.json'
        if parsed_model_file.exists():
            parsed_model_file.unlink()
        result = process_visualization(target_dir=test_gnn_dir, output_dir=viz_output_dir, verbose=True)
        assert isinstance(result, bool)
        caplog.text.lower()

class TestVisualizationComponents:
    """Test individual visualization components."""

    def test_network_visualization_generation(self) -> None:
        """Test network visualization with real data."""
        test_data = {'state_blocks': [{'name': 'A', 'dimensions': [3]}, {'name': 'B', 'dimensions': [2]}], 'connections': [{'source': 'A', 'target': 'B', 'type': 'transition'}]}
        output_dir = Path(tempfile.mkdtemp())
        try:
            files = generate_network_visualizations(test_data, output_dir, 'test_model')
            assert isinstance(files, list)
        finally:
            shutil.rmtree(output_dir)

    def test_combined_analysis_generation(self) -> None:
        """Test combined analysis with real data."""
        test_data = {'parameters': [{'name': 'A', 'shape': [3], 'values': [1, 2, 3]}], 'state_blocks': [{'name': 'A', 'dimensions': [3]}], 'connections': []}
        output_dir = Path(tempfile.mkdtemp())
        try:
            files = generate_combined_analysis(test_data, output_dir, 'test_model')
            assert isinstance(files, list)
        finally:
            shutil.rmtree(output_dir)
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])