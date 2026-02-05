#!/usr/bin/env python3
"""
Test Visualization Matrices - Specialized tests for matrix visualization functionality.

Tests the MatrixVisualizer class and matrix-specific visualization features.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMatrixVisualizerCore:
    """Core tests for MatrixVisualizer class."""

    @pytest.mark.fast
    def test_matrix_visualizer_instantiation(self):
        """Test MatrixVisualizer can be instantiated."""
        from visualization.matrix_visualizer import MatrixVisualizer
        
        visualizer = MatrixVisualizer()
        assert visualizer is not None

    @pytest.mark.fast
    def test_extract_matrix_data_from_parameters(self):
        """Test extraction of matrix data from GNN parameters."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        # Test with list of parameter dicts
        parameters = [
            {"name": "A", "value": [[0.5, 0.5], [0.3, 0.7]]},
            {"name": "B", "value": [[1, 0], [0, 1]]}
        ]
        
        matrices = visualizer.extract_matrix_data_from_parameters(parameters)
        
        assert isinstance(matrices, dict)
        assert "A" in matrices
        assert "B" in matrices
        assert isinstance(matrices["A"], np.ndarray)
        assert matrices["A"].shape == (2, 2)

    @pytest.mark.fast
    def test_extract_matrix_from_dict_format(self):
        """Test extraction when parameters are in dict format."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        # Dict format (name -> value mapping)
        parameters = {
            "transition": [[0.9, 0.1], [0.2, 0.8]],
            "observation": [[1, 0, 0], [0, 1, 0]]
        }
        
        matrices = visualizer.extract_matrix_data_from_parameters(parameters)
        
        assert "transition" in matrices
        assert matrices["transition"].shape == (2, 2)
        assert matrices["observation"].shape == (2, 3)

    @pytest.mark.fast
    def test_convert_to_matrix_1d(self):
        """Test conversion of 1D vectors."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        vector = [0.1, 0.2, 0.3, 0.4]
        matrix = visualizer._convert_to_matrix(vector, "test_vector")
        
        assert matrix is not None
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (4,)

    @pytest.mark.fast
    def test_convert_to_matrix_2d(self):
        """Test conversion of 2D matrices."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        matrix_data = [[1, 2, 3], [4, 5, 6]]
        matrix = visualizer._convert_to_matrix(matrix_data, "test_matrix")
        
        assert matrix is not None
        assert matrix.shape == (2, 3)

    @pytest.mark.fast
    def test_convert_to_matrix_3d(self):
        """Test conversion of 3D tensors."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        tensor_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        tensor = visualizer._convert_to_matrix(tensor_data, "test_tensor")
        
        assert tensor is not None
        assert tensor.shape == (2, 2, 2)


class TestMatrixHeatmapGeneration:
    """Tests for heatmap generation functionality."""

    @pytest.mark.slow
    def test_generate_matrix_heatmap(self, tmp_path):
        """Test heatmap generation for a 2D matrix."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        output_path = tmp_path / "heatmap.png"
        
        result = visualizer.generate_matrix_heatmap(
            matrix_name="test_matrix",
            matrix=matrix,
            output_path=output_path
        )
        
        assert result is True
        assert output_path.exists()

    @pytest.mark.slow
    def test_generate_heatmap_with_custom_colormap(self, tmp_path):
        """Test heatmap generation with custom colormap."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        matrix = np.random.rand(4, 4)
        output_path = tmp_path / "heatmap_custom.png"
        
        result = visualizer.generate_matrix_heatmap(
            matrix_name="random_matrix",
            matrix=matrix,
            output_path=output_path,
            cmap='Blues'
        )
        
        assert result is True
        assert output_path.exists()

    @pytest.mark.slow
    def test_create_heatmap_convenience_method(self, tmp_path, monkeypatch):
        """Test the create_heatmap convenience method."""
        from visualization.matrix_visualizer import MatrixVisualizer
        
        # Change working directory to tmp_path for output
        monkeypatch.chdir(tmp_path)
        (tmp_path / "output" / "2_tests_output").mkdir(parents=True, exist_ok=True)
        
        visualizer = MatrixVisualizer()
        
        matrix = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.4, 0.2]]
        result = visualizer.create_heatmap(matrix)
        
        assert result is True


class TestTensorVisualization:
    """Tests for 3D tensor visualization."""

    @pytest.mark.slow
    def test_generate_3d_tensor_visualization(self, tmp_path):
        """Test visualization of 3D POMDP transition matrices."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        # Create a 3D transition tensor (states x states x actions)
        tensor = np.zeros((3, 3, 2))
        tensor[:, :, 0] = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        tensor[:, :, 1] = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
        
        output_path = tmp_path / "tensor_viz.png"
        
        result = visualizer.generate_3d_tensor_visualization(
            tensor_name="B",
            tensor=tensor,
            output_path=output_path,
            tensor_type="transition"
        )
        
        assert result is True
        assert output_path.exists()

    @pytest.mark.slow
    def test_generate_pomdp_transition_analysis(self, tmp_path):
        """Test comprehensive POMDP transition analysis."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        # Create valid stochastic transition matrices
        tensor = np.zeros((3, 3, 2))
        tensor[:, :, 0] = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
        tensor[:, :, 1] = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        
        output_path = tmp_path / "pomdp_analysis.png"
        
        result = visualizer.generate_pomdp_transition_analysis(
            tensor=tensor,
            output_path=output_path
        )
        
        assert result is True
        assert output_path.exists()


class TestMatrixExportFormats:
    """Tests for matrix export to different formats."""

    @pytest.mark.fast
    def test_export_matrix_to_csv(self, tmp_path):
        """Test CSV export of matrix data."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output_path = tmp_path / "matrix.png"  # CSV will be matrix.csv
        
        result = visualizer.export_matrix_to_csv(
            matrix=matrix,
            matrix_name="test_matrix",
            output_path=output_path
        )
        
        assert result is True
        csv_path = output_path.with_suffix('.csv')
        assert csv_path.exists()
        
        # Verify CSV content
        content = csv_path.read_text()
        assert "test_matrix" in content
        assert "2, 3" in content  # Shape info

    @pytest.mark.fast
    def test_export_3d_tensor_to_csv(self, tmp_path):
        """Test CSV export of 3D tensor (first slice)."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        tensor = np.ones((2, 3, 4))
        output_path = tmp_path / "tensor.png"
        
        result = visualizer.export_matrix_to_csv(
            matrix=tensor,
            matrix_name="test_tensor",
            output_path=output_path
        )
        
        assert result is True


class TestMatrixStatistics:
    """Tests for matrix statistical analysis."""

    @pytest.mark.fast
    def test_tensor_statistics_generation(self):
        """Test generation of tensor statistics string."""
        from visualization.matrix_visualizer import MatrixVisualizer
        import numpy as np
        
        visualizer = MatrixVisualizer()
        
        # Valid transition matrix
        tensor = np.zeros((3, 3, 2))
        tensor[:, :, 0] = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
        tensor[:, :, 1] = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        
        stats = visualizer._generate_tensor_statistics(
            tensor=tensor,
            tensor_name="B",
            tensor_type="transition"
        )
        
        assert isinstance(stats, str)
        assert "B" in stats
        assert "Shape" in stats
        assert "Mean" in stats


class TestParsedGNNExtraction:
    """Tests for matrix extraction from parsed GNN data."""

    @pytest.mark.fast
    def test_extract_from_parsed_gnn_parameters(self):
        """Test extraction from GNN with parameters field."""
        from visualization.matrix_visualizer import MatrixVisualizer
        
        visualizer = MatrixVisualizer()
        
        parsed_data = {
            "ModelName": "TestModel",
            "parameters": [
                {"name": "A", "value": [[0.9, 0.1], [0.1, 0.9]]},
                {"name": "D", "value": [0.5, 0.5]}
            ]
        }
        
        matrices = visualizer.extract_from_parsed_gnn(parsed_data)
        
        assert "A" in matrices
        assert "D" in matrices

    @pytest.mark.fast
    def test_extract_from_initial_parameterization(self):
        """Test extraction from InitialParameterization section."""
        from visualization.matrix_visualizer import MatrixVisualizer
        
        visualizer = MatrixVisualizer()
        
        parsed_data = {
            "ModelName": "TestModel",
            "InitialParameterization": {
                "prior": [[0.3, 0.4, 0.3]],
                "likelihood": [[1, 0], [0, 1], [0.5, 0.5]]
            }
        }
        
        matrices = visualizer.extract_from_parsed_gnn(parsed_data)
        
        assert "prior" in matrices
        assert "likelihood" in matrices
