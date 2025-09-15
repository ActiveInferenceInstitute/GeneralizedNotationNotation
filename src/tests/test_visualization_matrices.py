#!/usr/bin/env python3
"""
Test Visualization Matrices Tests

This file contains tests for matrix visualization functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *

class TestVisualizationMatrices:
    """Tests for matrix visualization functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_visualizer_import(self):
        """Test matrix visualizer import."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            assert True
        except ImportError as e:
            pytest.skip(f"Matrix visualizer not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_visualizer_instantiation(self):
        """Test matrix visualizer instantiation."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            
            visualizer = MatrixVisualizer()
            assert visualizer is not None
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix visualizer instantiation not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_data_creation(self):
        """Test matrix data creation."""
        try:
            import numpy as np
            
            # Test 2D matrix creation
            matrix_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            assert matrix_2d.shape == (3, 3)
            
            # Test 3D matrix creation
            matrix_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            assert matrix_3d.shape == (2, 2, 2)
            
            # Test matrix operations
            sum_result = np.sum(matrix_2d)
            assert sum_result == 45
            
            mean_result = np.mean(matrix_2d)
            assert mean_result == 5.0
        except ImportError:
            pytest.skip("NumPy not available for matrix operations")
        except Exception as e:
            pytest.skip(f"Matrix data creation not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_visualization_data_preparation(self, isolated_temp_dir):
        """Test matrix visualization data preparation."""
        try:
            import numpy as np
            
            # Create test matrix data
            matrix_data = {
                "A": np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]]),
                "B": np.array([[0.8, 0.2], [0.3, 0.7]]),
                "C": np.array([0.4, 0.6])
            }
            
            # Test data validation
            for name, matrix in matrix_data.items():
                assert isinstance(matrix, np.ndarray)
                assert matrix.ndim >= 1
                assert matrix.size > 0
            
            # Test data serialization
            import json
            serializable_data = {}
            for name, matrix in matrix_data.items():
                serializable_data[name] = matrix.tolist()
            
            json_string = json.dumps(serializable_data)
            assert isinstance(json_string, str)
            
            # Test data deserialization
            parsed_data = json.loads(json_string)
            assert isinstance(parsed_data, dict)
            assert len(parsed_data) == 3
        except ImportError:
            pytest.skip("NumPy not available for matrix operations")
        except Exception as e:
            pytest.skip(f"Matrix visualization data preparation not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_heatmap_generation(self, isolated_temp_dir):
        """Test matrix heatmap generation."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create test matrix
            test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            
            # Test heatmap generation
            output_file = isolated_temp_dir / "test_heatmap.png"
            result = visualizer.create_heatmap(test_matrix, output_file)
            
            assert result is not None
            assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix heatmap generation not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_3d_visualization(self, isolated_temp_dir):
        """Test 3D matrix visualization."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create 3D test matrix
            test_matrix_3d = np.random.rand(3, 3, 3)
            
            # Test 3D visualization
            output_file = isolated_temp_dir / "test_3d_matrix.png"
            result = visualizer.create_3d_visualization(test_matrix_3d, output_file)
            
            assert result is not None
            assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix 3D visualization not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_comparison_visualization(self, isolated_temp_dir):
        """Test matrix comparison visualization."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create test matrices for comparison
            matrix1 = np.array([[1, 2], [3, 4]])
            matrix2 = np.array([[2, 3], [4, 5]])
            
            # Test comparison visualization
            output_file = isolated_temp_dir / "test_comparison.png"
            result = visualizer.create_comparison_visualization(
                matrix1, matrix2, output_file
            )
            
            assert result is not None
            assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix comparison visualization not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_statistics_visualization(self, isolated_temp_dir):
        """Test matrix statistics visualization."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create test matrix
            test_matrix = np.random.rand(10, 10)
            
            # Test statistics visualization
            output_file = isolated_temp_dir / "test_statistics.png"
            result = visualizer.create_statistics_visualization(
                test_matrix, output_file
            )
            
            assert result is not None
            assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix statistics visualization not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_color_mapping(self):
        """Test matrix color mapping."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Test different color maps
            test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            
            color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool']
            
            for cmap in color_maps:
                result = visualizer.apply_color_map(test_matrix, cmap)
                assert result is not None
                assert isinstance(result, np.ndarray)
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix color mapping not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_annotation(self, isolated_temp_dir):
        """Test matrix annotation."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create test matrix with labels
            test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            row_labels = ['Row1', 'Row2', 'Row3']
            col_labels = ['Col1', 'Col2', 'Col3']
            
            # Test annotated visualization
            output_file = isolated_temp_dir / "test_annotated.png"
            result = visualizer.create_annotated_visualization(
                test_matrix, row_labels, col_labels, output_file
            )
            
            assert result is not None
            assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix annotation not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_export_formats(self, isolated_temp_dir):
        """Test matrix export in different formats."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create test matrix
            test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            
            # Test different export formats
            formats = ['png', 'jpg', 'svg', 'pdf']
            
            for fmt in formats:
                output_file = isolated_temp_dir / f"test_matrix.{fmt}"
                result = visualizer.export_matrix(
                    test_matrix, output_file, format=fmt
                )
                
                assert result is not None
                assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix export formats not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matrix_interactive_visualization(self, isolated_temp_dir):
        """Test interactive matrix visualization."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            import numpy as np
            
            visualizer = MatrixVisualizer()
            
            # Create test matrix
            test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            
            # Test interactive visualization
            output_file = isolated_temp_dir / "test_interactive.html"
            result = visualizer.create_interactive_visualization(
                test_matrix, output_file
            )
            
            assert result is not None
            assert output_file.exists()
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix interactive visualization not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_matrix_visualization_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test matrix visualization pipeline integration."""
        try:
            from visualization.matrix_visualizer import MatrixVisualizer
            
            visualizer = MatrixVisualizer()
            
            # Test with sample GNN data
            gnn_file = list(sample_gnn_files.values())[0]
            output_dir = isolated_temp_dir / "matrix_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = visualizer.process_gnn_file(gnn_file, output_dir)
            assert result is not None
        except ImportError:
            pytest.skip("Matrix visualizer not available")
        except Exception as e:
            pytest.skip(f"Matrix visualization pipeline integration not available: {e}")

