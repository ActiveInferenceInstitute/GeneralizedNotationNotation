"""
Test suite for visualization functionality.

This module tests the matrix and ontology visualization components using real
test data and functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import shutil
import tempfile
import os

# Test markers
pytestmark = [pytest.mark.visualization, pytest.mark.safe_to_fail, pytest.mark.fast]

# Import visualization modules with error handling
try:
    from visualization.matrix_visualizer import MatrixVisualizer
    MATRIX_VISUALIZER_AVAILABLE = True
except ImportError:
    MATRIX_VISUALIZER_AVAILABLE = False

try:
    from visualization.ontology_visualizer import OntologyVisualizer
    ONTOLOGY_VISUALIZER_AVAILABLE = True
except ImportError:
    ONTOLOGY_VISUALIZER_AVAILABLE = False

try:
    import visualization
    from visualization import (
        create_graph_visualization,
        create_matrix_visualization,
        visualize_gnn_file,
        visualize_gnn_directory
    )
    VISUALIZATION_FUNCTIONS_AVAILABLE = True
except ImportError:
    VISUALIZATION_FUNCTIONS_AVAILABLE = False

@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory."""
    return Path(__file__).parent / 'test_data'

@pytest.fixture
def sample_gnn_file(test_data_dir):
    """Fixture providing path to sample GNN file."""
    return test_data_dir / 'sample_gnn_model.md'

@pytest.fixture
def temp_output_dir():
    """Fixture providing temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

class TestMatrixVisualizer:
    """Test cases for the MatrixVisualizer class."""
    
    @pytest.mark.skipif(not MATRIX_VISUALIZER_AVAILABLE, reason="MatrixVisualizer not available")
    def test_extract_matrices_from_content(self, sample_gnn_file):
        """Test matrix extraction from GNN file content."""
        # Read test file
        with open(sample_gnn_file, 'r') as f:
            content = f.read()
        
        # Extract matrices
        visualizer = MatrixVisualizer()
        matrices = visualizer._extract_matrices_from_content(content)
        
        # Verify extracted matrices
        assert len(matrices) == 5  # A, B, C, D, E matrices
        assert 'A' in matrices
        assert 'B' in matrices
        assert 'C' in matrices
        assert 'D' in matrices
        assert 'E' in matrices
        
        # Verify matrix dimensions
        assert len(matrices['A']) == 3  # 3x3 matrix
        assert len(matrices['A'][0]) == 3
        assert len(matrices['B']) == 3  # First slice of 3D matrix (3x3)
        assert len(matrices['B'][0]) == 3
        assert len(matrices['C']) == 1  # 1x3 matrix
        assert len(matrices['C'][0]) == 3
        assert len(matrices['D']) == 1  # 1x3 matrix
        assert len(matrices['D'][0]) == 3
        assert len(matrices['E']) == 1  # 1x3 matrix
        assert len(matrices['E'][0]) == 3
        
        # Verify some matrix values
        assert matrices['A'][0][0] == 0.9  # First element of A
        assert matrices['B'][0][0] == 1.0  # First element of first slice of B (identity matrix)
        assert matrices['B'][0][1] == 0.0  # Second element of first slice of B
        assert matrices['C'][0][2] == 1.0  # Last element of C
        assert matrices['D'][0][0] == 0.33333  # First element of D
        assert matrices['E'][0][0] == 0.33333  # First element of E
    
    def test_create_heatmap(self, temp_output_dir):
        """Test heatmap creation for a single matrix."""
        visualizer = MatrixVisualizer()
        matrix_data = [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9]
        ]
        
        # Create heatmap
        output_path = visualizer.create_heatmap('test_matrix', matrix_data, temp_output_dir)
        
        # Verify output
        assert output_path is not None
        assert Path(output_path).exists()
        assert Path(output_path).is_file()
        assert Path(output_path).suffix == '.png'
    
    def test_create_combined_matrix_visualization(self, temp_output_dir):
        """Test combined visualization of multiple matrices."""
        visualizer = MatrixVisualizer()
        matrices = {
            'A': [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9]
            ],
            'B': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            'C': [[0.1, 0.1, 1.0]]
        }
        
        # Create combined visualization
        output_path = visualizer.create_combined_matrix_visualization(matrices, temp_output_dir)
        
        # Verify output
        assert output_path is not None
        assert Path(output_path).exists()
        assert Path(output_path).is_file()
        assert Path(output_path).suffix == '.png'
    
    def test_visualize_directory(self, test_data_dir, temp_output_dir):
        """Test visualization of all matrices in a directory."""
        visualizer = MatrixVisualizer()
        output_files = visualizer.visualize_directory(test_data_dir, temp_output_dir)
        
        # Verify outputs
        assert len(output_files) > 0
        for file_path in output_files:
            assert Path(file_path).exists()
            assert Path(file_path).is_file()
            assert Path(file_path).suffix == '.png'

class TestOntologyVisualizer:
    """Test cases for the OntologyVisualizer class."""
    
    def test_extract_ontology_mappings(self, sample_gnn_file):
        """Test ontology mapping extraction from GNN file."""
        # Read test file
        with open(sample_gnn_file, 'r') as f:
            content = f.read()
        
        # Extract ontology section
        import re
        ontology_match = re.search(r'## ActInfOntologyAnnotation\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        assert ontology_match is not None
        
        # Extract mappings
        visualizer = OntologyVisualizer()
        mappings = visualizer._extract_ontology_mappings(ontology_match.group(1))
        
        # Verify mappings
        assert len(mappings) == 13  # All variables and matrices
        
        # Verify matrix mappings
        assert ('A', 'LikelihoodMatrix') in mappings
        assert ('B', 'TransitionMatrix') in mappings
        assert ('C', 'LogPreferenceVector') in mappings
        assert ('D', 'PriorOverHiddenStates') in mappings
        assert ('E', 'Habit') in mappings
        assert ('F', 'VariationalFreeEnergy') in mappings
        assert ('G', 'ExpectedFreeEnergy') in mappings
        
        # Verify variable mappings
        assert ('s', 'HiddenState') in mappings
        assert ('s_prime', 'NextHiddenState') in mappings
        assert ('o', 'Observation') in mappings
        assert ('π', 'PolicyVector') in mappings  # Distribution over actions
        assert ('u', 'Action') in mappings  # Chosen action
        assert ('t', 'Time') in mappings
    
    def test_create_ontology_table(self, temp_output_dir):
        """Test creation of ontology visualization table."""
        visualizer = OntologyVisualizer()
        mappings = [
            ('s1', 'state_location_1'),
            ('s2', 'state_location_2'),
            ('a1', 'action_move_forward'),
            ('o1', 'observation_position')
        ]
        
        # Create visualization
        output_path = visualizer._create_ontology_table(mappings, temp_output_dir)
        
        # Verify output
        assert output_path is not None
        assert Path(output_path).exists()
        assert Path(output_path).is_file()
        assert Path(output_path).suffix == '.png'
    
    def test_visualize_directory(self, test_data_dir, temp_output_dir):
        """Test visualization of all ontology annotations in a directory."""
        visualizer = OntologyVisualizer()
        output_files = visualizer.visualize_directory(test_data_dir, temp_output_dir)
        
        # Verify outputs
        assert len(output_files) > 0
        for file_path in output_files:
            assert Path(file_path).exists()
            assert Path(file_path).is_file()
            assert Path(file_path).suffix == '.png'

class TestVisualizationModule:
    """Test cases for the visualization module functions."""
    
    def test_create_graph_visualization(self, temp_output_dir):
        """Test graph visualization creation."""
        data = {
            'variables': ['s1', 's2', 's3', 'a1', 'a2', 'o1', 'o2'],
            'connections': [
                ('s1', 'o1'), ('s2', 'o1'), ('s3', 'o1'),
                ('s1', 's2'), ('s2', 's3'), ('s3', 's1'),
                ('a1', 's2'), ('a2', 's3')
            ]
        }
        output_path = temp_output_dir / 'graph.png'
        
        # Create visualization
        result_path = create_graph_visualization(data, output_path)
        
        # Verify output
        assert result_path is not None
        assert Path(result_path).exists()
        assert Path(result_path).is_file()
        assert Path(result_path).suffix == '.png'
    
    def test_create_matrix_visualization(self, temp_output_dir):
        """Test matrix visualization creation."""
        matrix = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.3, 0.6]
        ]
        output_path = temp_output_dir / 'matrix.png'
        
        # Create visualization
        result_path = create_matrix_visualization(matrix, output_path)
        
        # Verify output
        assert result_path is not None
        assert Path(result_path).exists()
        assert Path(result_path).is_file()
        assert Path(result_path).suffix == '.png'
    
    def test_visualize_gnn_file(self, sample_gnn_file, temp_output_dir):
        """Test visualization of a single GNN file."""
        results = visualize_gnn_file(sample_gnn_file, temp_output_dir)
        
        # Verify results structure
        assert 'matrices' in results
        assert 'ontology' in results
        assert 'graphs' in results
        
        # Verify matrix visualizations
        assert len(results['matrices']) > 0
        for file_path in results['matrices']:
            assert Path(file_path).exists()
            assert Path(file_path).is_file()
            assert Path(file_path).suffix == '.png'
        
        # Verify ontology visualizations
        assert len(results['ontology']) > 0
        for file_path in results['ontology']:
            assert Path(file_path).exists()
            assert Path(file_path).is_file()
            assert Path(file_path).suffix == '.png'
    
    def test_visualize_gnn_directory(self, test_data_dir, temp_output_dir):
        """Test visualization of all GNN files in a directory."""
        results = visualize_gnn_directory(test_data_dir, temp_output_dir)
        
        # Verify results
        assert results['total_files'] > 0
        assert results['matrix_visualizations'] > 0
        assert results['ontology_visualizations'] > 0
        
        # Verify output files exist
        output_files = list(temp_output_dir.glob('**/*.png'))
        assert len(output_files) > 0
        for file_path in output_files:
            assert file_path.exists()
            assert file_path.is_file()
            assert file_path.suffix == '.png' 