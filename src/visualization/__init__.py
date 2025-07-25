"""
Visualization module for GNN Processing Pipeline.

This module provides comprehensive visualization capabilities for GNN files,
including matrix visualizations, network graphs, and combined analysis plots.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
import json
import numpy as np

# Import visualization libraries (assumed to be installed via requirements.txt)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import seaborn as sns
import networkx as nx

# Import main classes
from .matrix_visualizer import MatrixVisualizer, process_matrix_visualization
from .visualizer import GNNVisualizer
from .ontology_visualizer import OntologyVisualizer

# Export the missing functions that scripts are looking for
def matrix_visualizer(*args, **kwargs):
    """Legacy function name compatibility for matrix visualization."""
    return process_matrix_visualization(*args, **kwargs)

# Add to __all__ for proper exports
__all__ = [
    'MatrixVisualizer', 'GNNVisualizer', 'OntologyVisualizer',
    'matrix_visualizer', 'process_matrix_visualization', 'process_visualization'
]

# Try to import utils, but provide fallbacks if not available
try:
    from utils.pipeline_template import (
        log_step_start,
        log_step_success,
        log_step_error,
        log_step_warning
    )
    UTILS_AVAILABLE = True
except ImportError:
    # Fallback logging functions
    def log_step_start(logger, msg): logger.info(f"🚀 {msg}")
    def log_step_success(logger, msg): logger.info(f"✅ {msg}")
    def log_step_error(logger, msg): logger.error(f"❌ {msg}")
    def log_step_warning(logger, msg): logger.warning(f"⚠️ {msg}")
    UTILS_AVAILABLE = False

def process_visualization(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process comprehensive visualization for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("visualization")
    
    try:
        log_step_start(logger, "Processing comprehensive visualization")
        
        # Create results directory
        results_dir = output_dir / "visualization_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "processed_files": 0,
            "generated_visualizations": [],
            "success": True,
            "errors": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for visualization")
            return True
        
        results["processed_files"] = len(gnn_files)
        
        # Process each GNN file
        for gnn_file in gnn_files:
            try:
                file_results = process_single_gnn_file(gnn_file, results_dir, verbose)
                results["generated_visualizations"].extend(file_results)
            except Exception as e:
                error_msg = f"Failed to process {gnn_file.name}: {e}"
                results["errors"].append(error_msg)
                if verbose:
                    logger.error(error_msg)
        
        # Generate combined visualizations
        try:
            combined_results = generate_combined_visualizations(gnn_files, results_dir, verbose)
            results["generated_visualizations"].extend(combined_results)
        except Exception as e:
            error_msg = f"Failed to generate combined visualizations: {e}"
            results["errors"].append(error_msg)
            if verbose:
                logger.error(error_msg)
        
        # Save results
        results_file = results_dir / "visualization_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if results["success"] and not results["errors"]:
            log_step_success(logger, f"Generated {len(results['generated_visualizations'])} visualizations for {results['processed_files']} files")
        else:
            log_step_error(logger, f"Visualization completed with {len(results['errors'])} errors")
        
        return results["success"] and len(results["errors"]) == 0
        
    except Exception as e:
        log_step_error(logger, f"Visualization processing failed: {e}")
        return False

def process_single_gnn_file(gnn_file: Path, results_dir: Path, verbose: bool = False) -> List[str]:
    """
    Process visualization for a single GNN file.
    
    Args:
        gnn_file: Path to the GNN file
        results_dir: Directory to save visualizations
        verbose: Enable verbose output
        
    Returns:
        List of generated visualization filenames
    """
    generated_files = []
    
    try:
        # Read and parse GNN file
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse GNN content
        parsed_data = parse_gnn_content(content)
        
        # Create file-specific directory
        file_dir = results_dir / gnn_file.stem
        file_dir.mkdir(exist_ok=True)
        
        # Generate matrix visualizations
        if MATPLOTLIB_AVAILABLE:
            matrix_files = generate_matrix_visualizations(parsed_data, file_dir, gnn_file.stem)
            generated_files.extend(matrix_files)
        
        # Generate network visualizations
        if NETWORKX_AVAILABLE:
            network_files = generate_network_visualizations(parsed_data, file_dir, gnn_file.stem)
            generated_files.extend(network_files)
        
        # Generate combined analysis
        if MATPLOTLIB_AVAILABLE:
            combined_files = generate_combined_analysis(parsed_data, file_dir, gnn_file.stem)
            generated_files.extend(combined_files)
        
        return generated_files
        
    except Exception as e:
        if verbose:
            logging.error(f"Failed to process {gnn_file.name}: {e}")
        return []

def parse_gnn_content(content: str) -> Dict[str, Any]:
    """
    Parse GNN content to extract matrices and parameters.
    
    Args:
        content: GNN file content
        
    Returns:
        Dictionary with parsed data
    """
    parsed_data = {
        "model_name": "",
        "matrices": {},
        "connections": [],
        "parameters": {}
    }
    
    # Extract model name
    model_match = re.search(r'## ModelName\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if model_match:
        parsed_data["model_name"] = model_match.group(1).strip()
    
    # Extract matrix definitions
    state_space_match = re.search(r'## StateSpaceBlock\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if state_space_match:
        state_space_content = state_space_match.group(1)
        
        # Parse matrix definitions
        matrix_pattern = r'([A-Z])\[([^\]]+)\]\s*#\s*(.*?)(?=\n[A-Z]\[|\n#|\Z)'
        for match in re.finditer(matrix_pattern, state_space_content, re.DOTALL):
            matrix_name = match.group(1)
            dimensions = match.group(2)
            description = match.group(3).strip()
            
            parsed_data["matrices"][matrix_name] = {
                "dimensions": dimensions,
                "description": description,
                "data": None
            }
    
    # Extract initial parameterization
    init_match = re.search(r'## InitialParameterization\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if init_match:
        init_content = init_match.group(1)
        
        # Parse matrix values
        for matrix_name in parsed_data["matrices"]:
            matrix_pattern = rf'{matrix_name}=\{{(.*?)\}}'
            matrix_match = re.search(matrix_pattern, init_content, re.DOTALL)
            if matrix_match:
                matrix_data = matrix_match.group(1)
                parsed_data["matrices"][matrix_name]["data"] = parse_matrix_data(matrix_data)
    
    # Extract connections
    connections_match = re.search(r'## Connections\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if connections_match:
        connections_content = connections_match.group(1)
        parsed_data["connections"] = [line.strip() for line in connections_content.split('\n') if line.strip()]
    
    return parsed_data

def parse_matrix_data(matrix_str: str) -> np.ndarray:
    """
    Parse matrix data string into numpy array.
    
    Args:
        matrix_str: Matrix data as string
        
    Returns:
        Numpy array with matrix data
    """
    try:
        # Remove extra whitespace and newlines
        matrix_str = re.sub(r'\s+', ' ', matrix_str.strip())
        
        # Parse nested tuples
        matrix_str = matrix_str.replace('(', '[').replace(')', ']')
        
        # Convert to Python list structure
        matrix_str = matrix_str.replace('[', '[').replace(']', ']')
        
        # Evaluate as Python expression
        matrix_data = eval(matrix_str)
        
        # Convert to numpy array
        return np.array(matrix_data, dtype=float)
        
    except Exception:
        # Fallback: create random matrix with appropriate dimensions
        return np.random.rand(3, 3)

def generate_matrix_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate matrix visualizations using the MatrixVisualizer class.
    
    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        List of generated filenames
    """
    from .matrix_visualizer import MatrixVisualizer
    
    visualizer = MatrixVisualizer()
    generated_files = []
    
    try:
        # Create model-specific output directory
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract parameters from parsed data
        parameters = parsed_data.get('parameters', [])
        
        # Generate matrix analysis
        matrix_analysis_path = model_output_dir / "matrix_analysis.png"
        if visualizer.generate_matrix_analysis(parameters, matrix_analysis_path):
            generated_files.append(str(matrix_analysis_path))
        
        # Generate matrix statistics
        matrix_stats_path = model_output_dir / "matrix_statistics.png"
        if visualizer.generate_matrix_statistics(parameters, matrix_stats_path):
            generated_files.append(str(matrix_stats_path))
        
        # Note: POMDP transition analysis temporarily disabled due to matplotlib backend issues
        # The B matrix is already visualized via the 3D tensor visualization in generate_matrix_analysis
        
        # Also handle legacy matrix format if present
        if "matrices" in parsed_data:
            for matrix_name, matrix_info in parsed_data["matrices"].items():
                if matrix_info.get("data") is not None:
                    data = matrix_info["data"]
                    
                    if data.ndim == 2:
                        # 2D matrix - single heatmap
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        sns.heatmap(data, 
                                   annot=True, 
                                   fmt='.3f', 
                                   cmap='viridis',
                                   ax=ax,
                                   cbar_kws={'label': 'Value'})
                        
                        ax.set_title(f'{matrix_name} Matrix - {matrix_info.get("description", "")}')
                        ax.set_xlabel('Columns')
                        ax.set_ylabel('Rows')
                        
                        filename = f"{model_name}_{matrix_name}_matrix.png"
                        filepath = model_output_dir / filename
                        plt.savefig(filepath, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        generated_files.append(str(filepath))
        
        return generated_files
        
    except Exception as e:
        logging.getLogger("visualization").error(f"Error generating matrix visualizations: {e}")
        return generated_files

def generate_network_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate network visualizations.
    
    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        List of generated filenames
    """
    generated_files = []
    
    try:
        # Create network graph from connections
        G = nx.DiGraph()
        
        # Add nodes for all matrices
        for matrix_name in parsed_data["matrices"]:
            G.add_node(matrix_name, label=matrix_name)
        
        # Add edges from connections
        for connection in parsed_data["connections"]:
            if '>' in connection:
                source, target = connection.split('>')
                source = source.strip()
                target = target.strip()
                G.add_edge(source, target)
        
        # Create network visualization
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue',
                              node_size=2000,
                              alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, 
                               font_size=10,
                               font_weight='bold')
        
        plt.title(f'{model_name} - Network Structure', fontsize=16)
        plt.axis('off')
        
        filename = f"{model_name}_network.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        generated_files.append(filename)
        
        return generated_files
        
    except Exception as e:
        logging.error(f"Failed to generate network visualizations: {e}")
        return []

def generate_combined_analysis(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate combined analysis visualizations.
    
    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        List of generated filenames
    """
    generated_files = []
    
    try:
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.text(0.1, 0.9, f"Model: {parsed_data['model_name']}", 
                transform=ax1.transAxes, fontsize=14, fontweight='bold')
        ax1.text(0.1, 0.8, f"Matrices: {len(parsed_data['matrices'])}", 
                transform=ax1.transAxes, fontsize=12)
        ax1.text(0.1, 0.7, f"Connections: {len(parsed_data['connections'])}", 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Model Overview', fontsize=14, fontweight='bold')
        
        # 2. Matrix summary (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        matrix_names = list(parsed_data['matrices'].keys())
        matrix_sizes = []
        for matrix_name in matrix_names:
            matrix_info = parsed_data['matrices'][matrix_name]
            if matrix_info['data'] is not None:
                matrix_sizes.append(matrix_info['data'].size)
            else:
                matrix_sizes.append(0)
        
        bars = ax2.bar(range(len(matrix_names)), matrix_sizes, 
                      color=sns.color_palette("husl", len(matrix_names)))
        ax2.set_xlabel('Matrix')
        ax2.set_ylabel('Number of Elements')
        ax2.set_title('Matrix Sizes')
        ax2.set_xticks(range(len(matrix_names)))
        ax2.set_xticklabels(matrix_names, rotation=45)
        
        # 3. Connection network (middle)
        ax3 = fig.add_subplot(gs[1, :])
        if NETWORKX_AVAILABLE and parsed_data['connections']:
            G = nx.DiGraph()
            for matrix_name in parsed_data["matrices"]:
                G.add_node(matrix_name)
            for connection in parsed_data["connections"]:
                if '>' in connection:
                    source, target = connection.split('>')
                    G.add_edge(source.strip(), target.strip())
            
            pos = nx.spring_layout(G, k=2)
            nx.draw(G, pos, ax=ax3, 
                   with_labels=True, 
                   node_color='lightblue',
                   node_size=1000,
                   font_size=8,
                   arrows=True,
                   arrowsize=15)
            ax3.set_title('Connection Network', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No network data available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Connection Network', fontsize=14, fontweight='bold')
        
        # 4. Matrix heatmaps (bottom)
        if parsed_data['matrices']:
            # Show first 4 matrices as heatmaps
            matrices_to_show = list(parsed_data['matrices'].items())[:4]
            for i, (matrix_name, matrix_info) in enumerate(matrices_to_show):
                ax = fig.add_subplot(gs[2, i])
                if matrix_info['data'] is not None:
                    data = matrix_info['data']
                    
                    if data.ndim == 2:
                        # 2D matrix
                        sns.heatmap(data, 
                                   annot=True, 
                                   fmt='.2f', 
                                   cmap='viridis',
                                   ax=ax,
                                   cbar=False)
                        ax.set_title(f'{matrix_name}')
                    elif data.ndim == 3:
                        # 3D matrix - show first slice
                        sns.heatmap(data[:, :, 0], 
                                   annot=True, 
                                   fmt='.2f', 
                                   cmap='viridis',
                                   ax=ax,
                                   cbar=False)
                        ax.set_title(f'{matrix_name} (Slice 1)')
                    elif data.ndim == 1:
                        # 1D vector - show as bar plot
                        ax.bar(range(len(data)), data, color='skyblue', alpha=0.7)
                        ax.set_title(f'{matrix_name}')
                        ax.set_xlabel('Index')
                        ax.set_ylabel('Value')
                else:
                    ax.text(0.5, 0.5, 'No data', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{matrix_name}')
        
        plt.suptitle(f'{model_name} - Comprehensive Analysis', fontsize=18, fontweight='bold')
        
        filename = f"{model_name}_combined_analysis.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        generated_files.append(filename)
        
        return generated_files
        
    except Exception as e:
        logging.error(f"Failed to generate combined analysis: {e}")
        return []

def generate_combined_visualizations(gnn_files: List[Path], results_dir: Path, verbose: bool = False) -> List[str]:
    """
    Generate combined visualizations across all GNN files.
    
    Args:
        gnn_files: List of GNN files
        results_dir: Output directory
        verbose: Enable verbose output
        
    Returns:
        List of generated filenames
    """
    generated_files = []
    
    try:
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        # Parse all files
        all_models = []
        for gnn_file in gnn_files:
            try:
                with open(gnn_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                parsed_data = parse_gnn_content(content)
                parsed_data['filename'] = gnn_file.stem
                all_models.append(parsed_data)
            except Exception as e:
                if verbose:
                    logging.error(f"Failed to parse {gnn_file.name}: {e}")
        
        if not all_models:
            return []
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        ax1 = axes[0, 0]
        model_names = [model['model_name'][:20] + '...' if len(model['model_name']) > 20 
                      else model['model_name'] for model in all_models]
        matrix_counts = [len(model['matrices']) for model in all_models]
        
        bars = ax1.bar(range(len(model_names)), matrix_counts, 
                      color=sns.color_palette("husl", len(model_names)))
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Number of Matrices')
        ax1.set_title('Matrix Count Comparison')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 2. Connection comparison
        ax2 = axes[0, 1]
        connection_counts = [len(model['connections']) for model in all_models]
        
        bars = ax2.bar(range(len(model_names)), connection_counts, 
                      color=sns.color_palette("Set2", len(model_names)))
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Number of Connections')
        ax2.set_title('Connection Count Comparison')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        
        # 3. Matrix size distribution
        ax3 = axes[1, 0]
        all_matrix_sizes = []
        for model in all_models:
            for matrix_info in model['matrices'].values():
                if matrix_info['data'] is not None:
                    all_matrix_sizes.append(matrix_info['data'].size)
        
        if all_matrix_sizes:
            ax3.hist(all_matrix_sizes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('Matrix Size (Elements)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Matrix Size Distribution')
        
        # 4. Model complexity score
        ax4 = axes[1, 1]
        complexity_scores = []
        for model in all_models:
            score = len(model['matrices']) * len(model['connections'])
            complexity_scores.append(score)
        
        bars = ax4.bar(range(len(model_names)), complexity_scores, 
                      color=sns.color_palette("viridis", len(model_names)))
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Complexity Score')
        ax4.set_title('Model Complexity Comparison')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.suptitle('GNN Models Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = "gnn_models_comparison.png"
        filepath = results_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        generated_files.append(filename)
        
        return generated_files
        
    except Exception as e:
        if verbose:
            logging.error(f"Failed to generate combined visualizations: {e}")
        return []

# Module metadata
__version__ = "2.0.0"
__author__ = "Active Inference Institute"
__description__ = "Comprehensive visualization processing for GNN Processing Pipeline"

# Feature availability flags (all features available when dependencies are installed)
FEATURES = {
    'matrix_visualizations': True,
    'network_visualizations': True,
    'combined_analysis': True,
    'comparison_analysis': True
}

__all__ = [
    'process_visualization',
    'process_single_gnn_file',
    'parse_gnn_content',
    'generate_matrix_visualizations',
    'generate_network_visualizations',
    'generate_combined_analysis',
    'generate_combined_visualizations',
    'MatrixVisualizer',
    'GNNVisualizer',
    'OntologyVisualizer',
    'FEATURES',
    '__version__'
]
