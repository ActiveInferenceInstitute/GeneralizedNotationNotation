"""
Visualization Module for GNN Pipeline

This module provides comprehensive visualization capabilities for GNN models,
including matrix visualizations, ontology visualizations, and graph representations.
All visualizations are created using real matplotlib and numpy functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Import visualizer classes
from .matrix_visualizer import MatrixVisualizer
from .ontology_visualizer import OntologyVisualizer

# Set up module logger
logger = logging.getLogger(__name__)

__all__ = [
    'MatrixVisualizer',
    'OntologyVisualizer',
    'create_graph_visualization',
    'create_matrix_visualization',
    'visualize_gnn_file',
    'visualize_gnn_directory'
]

def create_graph_visualization(data: Dict[str, Any], output_path: Path) -> Optional[str]:
    """
    Create a graph visualization from GNN data.
    
    Args:
        data: GNN model data containing graph structure
        output_path: Path where visualization should be saved
        
    Returns:
        Path to saved visualization file, or None if failed
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create a graph based on GNN data
        G = nx.Graph()
        
        # Add nodes and edges if available
        if 'variables' in data:
            G.add_nodes_from(data['variables'])
        
        if 'connections' in data:
            G.add_edges_from(data['connections'])
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=10)
        
        plt.title('GNN Model Graph Structure')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graph visualization saved to {output_path}")
        return str(output_path)
        
    except ImportError:
        logger.error("NetworkX not available for graph visualization")
        return None
    except Exception as e:
        logger.error(f"Failed to create graph visualization: {e}")
        return None

def create_matrix_visualization(matrix: List[List[float]], output_path: Path) -> Optional[str]:
    """
    Create a matrix visualization.
    
    Args:
        matrix: 2D list representing the matrix
        output_path: Path where visualization should be saved
        
    Returns:
        Path to saved visualization file, or None if failed
    """
    try:
        visualizer = MatrixVisualizer()
        return visualizer.create_heatmap("matrix", matrix, output_path.parent)
    except Exception as e:
        logger.error(f"Failed to create matrix visualization: {e}")
        return None

def visualize_gnn_file(gnn_file: Path, output_dir: Path) -> Dict[str, List[str]]:
    """
    Create all visualizations for a single GNN file.
    
    Args:
        gnn_file: Path to GNN file
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping visualization types to lists of created files
    """
    results = {
        'matrices': [],
        'ontology': [],
        'graphs': []
    }
    
    try:
        # Create output directory for this file
        file_output_dir = output_dir / gnn_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create matrix visualizations
        matrix_visualizer = MatrixVisualizer()
        matrix_files = matrix_visualizer.visualize_directory(gnn_file.parent, file_output_dir)
        results['matrices'] = matrix_files
        
        # Create ontology visualizations
        ontology_visualizer = OntologyVisualizer()
        ontology_files = ontology_visualizer.visualize_directory(gnn_file.parent, file_output_dir)
        results['ontology'] = ontology_files
        
        logger.info(f"Generated visualizations for {gnn_file.name}: "
                   f"{len(matrix_files)} matrix, {len(ontology_files)} ontology")
        
    except Exception as e:
        logger.error(f"Failed to visualize GNN file {gnn_file}: {e}")
    
    return results

def visualize_gnn_directory(input_dir: Path, output_dir: Path) -> Dict[str, int]:
    """
    Create visualizations for all GNN files in a directory.
    
    Args:
        input_dir: Directory containing GNN files
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with counts of created visualizations by type
    """
    results = {
        'total_files': 0,
        'matrix_visualizations': 0,
        'ontology_visualizations': 0,
        'graph_visualizations': 0
    }
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all GNN files
        gnn_files = list(input_dir.glob('**/*.md'))
        results['total_files'] = len(gnn_files)
        
        for gnn_file in gnn_files:
            file_results = visualize_gnn_file(gnn_file, output_dir)
            results['matrix_visualizations'] += len(file_results['matrices'])
            results['ontology_visualizations'] += len(file_results['ontology'])
            results['graph_visualizations'] += len(file_results['graphs'])
        
        logger.info(f"Processed {results['total_files']} GNN files, "
                   f"created {results['matrix_visualizations']} matrix, "
                   f"{results['ontology_visualizations']} ontology, "
                   f"{results['graph_visualizations']} graph visualizations")
        
    except Exception as e:
        logger.error(f"Failed to visualize GNN directory {input_dir}: {e}")
    
    return results 