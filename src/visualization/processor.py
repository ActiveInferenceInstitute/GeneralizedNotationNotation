#!/usr/bin/env python3
"""
Visualization processor module for GNN Processing Pipeline.

This module provides the main visualization processing functionality.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
import json
import numpy as np
import matplotlib
# Force non-interactive backend to avoid GUI/dpi hangs in headless environments
matplotlib.use('Agg')

# Import visualization libraries with error handling for testing
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RecursionError):
    plt = None
    patches = None
    cm = None
    MATPLOTLIB_AVAILABLE = False

# Seaborn is optional; fall back to matplotlib-only if unavailable
try:
    import seaborn as sns  # type: ignore
    SEABORN_AVAILABLE = True
except Exception:
    sns = None  # type: ignore
    SEABORN_AVAILABLE = False
    
# Safe NetworkX import to avoid pathlib recursion errors
try:
    import sys
    if sys.version_info >= (3, 13):
        # For Python 3.13+, use a safer import approach
        import os
        # Disable automatic backends completely for Python 3.13
        os.environ.pop('NETWORKX_AUTOMATIC_BACKENDS', None)
        os.environ['NETWORKX_CACHE_CONVERTED_GRAPHS'] = '1'
    import networkx as nx
    NETWORKX_AVAILABLE = True
except (ImportError, RecursionError, AttributeError, ValueError) as e:
    nx = None
    NETWORKX_AVAILABLE = False

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


# Safe plot saving helper to avoid crashes due to extreme DPI or backend issues
def _save_plot_safely(plot_path: Path, dpi: int = 300, **savefig_kwargs) -> bool:
    """Attempt to save a matplotlib figure with fallback DPI strategies.

    Returns True on success, False on failure.
    """
    try:
        plt.savefig(plot_path, dpi=dpi, **savefig_kwargs)
        return True
    except Exception:
        try:
            fallback_dpi = int(matplotlib.rcParams.get('savefig.dpi', 100))
            plt.savefig(plot_path, dpi=fallback_dpi, **savefig_kwargs)
            return True
        except Exception:
            try:
                plt.savefig(plot_path, **savefig_kwargs)
                return True
            except Exception:
                return False

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
        log_step_start(logger, "Processing visualizations")
        
        # Create results directory
        results_dir = output_dir / "visualization_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find GNN files (.md primary; support .gnn for tests)
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            gnn_files = list(target_dir.glob("*.gnn"))
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for visualization")
            return True
        
        # Process each GNN file
        all_visualizations = []
        for gnn_file in gnn_files:
            try:
                file_visualizations = process_single_gnn_file(gnn_file, results_dir, verbose)
                all_visualizations.extend(file_visualizations)
            except Exception as e:
                logger.error(f"Error processing {gnn_file}: {e}")
        
        # Generate combined visualizations
        if len(gnn_files) > 1:
            try:
                combined_viz = generate_combined_visualizations(gnn_files, results_dir, verbose)
                all_visualizations.extend(combined_viz)
            except Exception as e:
                logger.error(f"Error generating combined visualizations: {e}")
        
        # Save results summary
        results_summary = {
            "processed_files": len(gnn_files),
            "total_visualizations": len(all_visualizations),
            "visualization_files": all_visualizations,
            "success": len(all_visualizations) > 0
        }
        
        summary_file = results_dir / "visualization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        if results_summary["success"]:
            log_step_success(logger, f"Generated {len(all_visualizations)} visualizations")
        else:
            log_step_error(logger, "No visualizations generated")
        
        return results_summary["success"]
        
    except Exception as e:
        log_step_error(logger, f"Visualization processing failed: {e}")
        return False

def process_single_gnn_file(gnn_file: Path, results_dir: Path, verbose: bool = False) -> List[str]:
    """
    Process visualization for a single GNN file with performance optimization.
    
    Args:
        gnn_file: Path to the GNN file
        results_dir: Directory to save results
        verbose: Enable verbose output
        
    Returns:
        List of generated visualization file paths
    """
    try:
        # Read file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create model-specific output directory
        model_name = gnn_file.stem
        model_dir = results_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Enhanced caching: check for existing visualizations with timestamp
        existing_pngs = sorted([str(p) for p in model_dir.glob('*.png')])
        if existing_pngs:
            # Check if cache is still valid (file modification time)
            source_mtime = gnn_file.stat().st_mtime
            cache_mtime = min(Path(png).stat().st_mtime for png in existing_pngs)
            
            if cache_mtime >= source_mtime:
                if verbose:
                    print(f"Using cached visualizations for {model_name}")
                return existing_pngs
            else:
                # Cache is stale, remove old files
                for png_file in existing_pngs:
                    try:
                        Path(png_file).unlink()
                    except OSError:
                        pass
        
        # Parse GNN content with optimization checks
        parsed_data = parse_gnn_content(content)
        
        # Check for large datasets and apply sampling if needed
        should_sample = False
        if parsed_data.get("variables") and len(parsed_data["variables"]) > 100:
            should_sample = True
            if verbose:
                print(f"Large dataset detected for {model_name}, applying sampling")
        
        if should_sample:
            # Sample data to improve performance
            original_vars = len(parsed_data.get("variables", []))
            original_conns = len(parsed_data.get("connections", []))
            
            # Keep first 100 variables and related connections
            parsed_data["variables"] = parsed_data["variables"][:100]
            var_names = {var["name"] for var in parsed_data["variables"]}
            parsed_data["connections"] = [conn for conn in parsed_data.get("connections", [])
                                        if conn["source"] in var_names and conn["target"] in var_names]
            
            # Limit matrices
            if parsed_data.get("matrices") and len(parsed_data["matrices"]) > 5:
                parsed_data["matrices"] = parsed_data["matrices"][:5]
            
            parsed_data["_sampling_applied"] = {
                "original_variables": original_vars,
                "original_connections": original_conns,
                "sampled_variables": len(parsed_data["variables"]),
                "sampled_connections": len(parsed_data["connections"])
            }
        
        # Generate different types of visualizations with error handling
        visualizations = []
        
        # Matrix visualizations (skip if too many matrices)
        if len(parsed_data.get("matrices", [])) <= 10:
            try:
                matrix_viz = generate_matrix_visualizations(parsed_data, model_dir, model_name)
                visualizations.extend(matrix_viz)
            except Exception as e:
                if verbose:
                    print(f"Matrix visualization failed for {model_name}: {e}")
        elif verbose:
            print(f"Skipping matrix visualizations for {model_name} - too many matrices")
        
        # Network visualizations (skip if too many nodes)
        if len(parsed_data.get("variables", [])) <= 200:
            try:
                network_viz = generate_network_visualizations(parsed_data, model_dir, model_name)
                visualizations.extend(network_viz)
            except Exception as e:
                if verbose:
                    print(f"Network visualization failed for {model_name}: {e}")
        elif verbose:
            print(f"Skipping network visualizations for {model_name} - too many nodes")
        
        # Combined analysis (always try this as it's lightweight)
        try:
            combined_viz = generate_combined_analysis(parsed_data, model_dir, model_name)
            visualizations.extend(combined_viz)
        except Exception as e:
            if verbose:
                print(f"Combined analysis failed for {model_name}: {e}")
        
        # Add sampling note to visualizations if applied
        if should_sample and visualizations:
            try:
                # Create a note file about sampling
                sampling_note = model_dir / f"{model_name}_sampling_note.txt"
                with open(sampling_note, 'w') as f:
                    f.write(f"Sampling applied to {model_name}:\n")
                    f.write(f"Original variables: {parsed_data['_sampling_applied']['original_variables']}\n")
                    f.write(f"Sampled variables: {parsed_data['_sampling_applied']['sampled_variables']}\n")
                    f.write(f"Original connections: {parsed_data['_sampling_applied']['original_connections']}\n")
                    f.write(f"Sampled connections: {parsed_data['_sampling_applied']['sampled_connections']}\n")
            except Exception:
                pass
        
        return visualizations
        
    except Exception as e:
        raise Exception(f"Failed to process visualization for {gnn_file}: {e}")

def parse_gnn_content(content: str) -> Dict[str, Any]:
    """
    Parse GNN content into structured data for visualization.
    
    Args:
        content: Raw GNN file content
        
    Returns:
        Dictionary with parsed GNN data
    """
    try:
        parsed = {
            "sections": {},
            "variables": [],
            "connections": [],
            "matrices": [],
            "metadata": {}
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.startswith('#'):
                current_section = line.lstrip('#').strip()
                parsed["sections"][current_section] = []
            elif current_section:
                parsed["sections"][current_section].append(line)
                
                # Extract variables and connections
                if ':' in line and '=' not in line:
                    # Variable definition
                    var_parts = line.split(':', 1)
                    if len(var_parts) == 2:
                        parsed["variables"].append({
                            "name": var_parts[0].strip(),
                            "type": var_parts[1].strip()
                        })
                elif '->' in line or '→' in line:
                    # Connection definition
                    conn_parts = line.split('->' if '->' in line else '→', 1)
                    if len(conn_parts) == 2:
                        parsed["connections"].append({
                            "source": conn_parts[0].strip(),
                            "target": conn_parts[1].strip()
                        })
                elif '[' in line and ']' in line:
                    # Potential matrix definition
                    try:
                        matrix_data = parse_matrix_data(line)
                        if matrix_data is not None:
                            parsed["matrices"].append({
                                "data": matrix_data,
                                "definition": line
                            })
                    except:
                        pass
        
        return parsed
        
    except Exception as e:
        return {
            "error": str(e),
            "sections": {},
            "variables": [],
            "connections": [],
            "matrices": [],
            "metadata": {}
        }

def parse_matrix_data(matrix_str: str) -> np.ndarray:
    """
    Parse matrix data from string representation.
    
    Args:
        matrix_str: String representation of matrix
        
    Returns:
        Numpy array of matrix data
    """
    try:
        # Extract matrix content between brackets
        start = matrix_str.find('[')
        end = matrix_str.rfind(']')
        
        if start == -1 or end == -1:
            return None
        
        matrix_content = matrix_str[start+1:end]
        
        # Parse matrix rows
        rows = []
        for row_str in matrix_content.split(';'):
            row_str = row_str.strip()
            if row_str:
                row = [float(x.strip()) for x in row_str.split(',') if x.strip()]
                rows.append(row)
        
        if rows:
            return np.array(rows)
        else:
            return None
            
    except Exception:
        return None

def generate_matrix_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate matrix visualizations.
    
    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        List of generated visualization file paths
    """
    visualizations = []
    
    if not MATPLOTLIB_AVAILABLE:
        return visualizations
    
    try:
        # Generate heatmaps for matrices with size optimization
        matrices = parsed_data.get("matrices", [])
        for i, matrix_info in enumerate(matrices):
            matrix_data = matrix_info["data"]
            
            if matrix_data is not None and matrix_data.size > 0:
                # Optimize large matrices for visualization
                original_shape = matrix_data.shape
                display_matrix = matrix_data
                
                # Sample large matrices to improve performance and readability
                max_display_size = 50
                if matrix_data.shape[0] > max_display_size or matrix_data.shape[1] > max_display_size:
                    # Sample rows and columns
                    row_step = max(1, matrix_data.shape[0] // max_display_size)
                    col_step = max(1, matrix_data.shape[1] // max_display_size)
                    display_matrix = matrix_data[::row_step, ::col_step]
                
                # Create heatmap with optimized settings
                fig_width = min(12, max(6, display_matrix.shape[1] * 0.3))
                fig_height = min(10, max(4, display_matrix.shape[0] * 0.3))
                plt.figure(figsize=(fig_width, fig_height))
                
                if SEABORN_AVAILABLE:
                    # Only annotate small matrices to avoid clutter
                    show_annot = display_matrix.size <= 100
                    sns.heatmap(display_matrix, annot=show_annot, cmap='viridis', 
                               fmt='.2f' if show_annot else '', cbar_kws={'shrink': 0.8})
                else:
                    im = plt.imshow(display_matrix, cmap='viridis', aspect='auto')
                    plt.colorbar(im, fraction=0.046, pad=0.04, shrink=0.8)
                    # Annotate very small matrices only
                    if display_matrix.size <= 25:
                        try:
                            rows, cols = display_matrix.shape
                            for r in range(rows):
                                for c in range(cols):
                                    plt.text(c, r, f"{display_matrix[r, c]:.2f}", 
                                           ha='center', va='center', color='white', fontsize=8)
                        except Exception:
                            pass
                
                # Add title with shape information
                title = f"{model_name} - Matrix {i+1}"
                if original_shape != display_matrix.shape:
                    title += f" (sampled from {original_shape[0]}x{original_shape[1]})"
                plt.title(title, fontsize=10)
                plt.tight_layout()

                # Save plot with optimized settings
                plot_file = output_dir / f"{model_name}_matrix_{i+1}_heatmap.png"
                # Reduce DPI for large matrices to save memory and disk space
                dpi = 150 if display_matrix.size > 1000 else 300
                _save_plot_safely(plot_file, dpi=dpi, bbox_inches='tight')
                plt.close()
                
                visualizations.append(str(plot_file))
        
        # Matrix-to-matrix correlation based on flattened numeric values (real data, no randomness)
        matrices_list = [m for m in matrices if m.get("data") is not None]
        if len(matrices_list) > 1:
            try:
                names = [m.get("definition", f"M{i+1}") for i, m in enumerate(matrices_list)]
                data_flat = []
                max_len = max(m["data"].size for m in matrices_list)
                for m in matrices_list:
                    flat = m["data"].astype(float).flatten()
                    # Pad with NaNs to equal length for fair correlation on overlapping indices
                    if flat.size < max_len:
                        pad = np.full((max_len - flat.size,), np.nan)
                        flat = np.concatenate([flat, pad])
                    data_flat.append(flat)
                data_mat = np.vstack(data_flat)
                # Compute pairwise correlations ignoring NaNs
                def nan_corrcoef(arr):
                    # arr shape: num_matrices x max_len
                    with np.errstate(invalid='ignore'):
                        mean = np.nanmean(arr, axis=1, keepdims=True)
                        std = np.nanstd(arr, axis=1, keepdims=True)
                        norm = (arr - mean) / (std + 1e-12)
                        cov = np.nanmean(norm[:, None, :] * norm[None, :, :], axis=2)
                        return cov
                corr_matrix = nan_corrcoef(data_mat)
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           xticklabels=[f"M{i+1}" for i in range(len(names))],
                           yticklabels=[f"M{i+1}" for i in range(len(names))],
                           annot=True if len(names) <= 10 else False, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f"{model_name} - Matrix Correlation (flattened)")
                plt.tight_layout()
                plot_file = output_dir / f"{model_name}_matrix_correlation.png"
                _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(plot_file))
            except Exception as _:
                pass
        
    except Exception as e:
        print(f"Error generating matrix visualizations: {e}")
    
    return visualizations

def generate_network_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate network visualizations.
    
    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        List of generated visualization file paths
    """
    visualizations = []
    
    if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
        return visualizations
    
    try:
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes (variables)
        variables = parsed_data.get("variables", [])
        for var in variables:
            G.add_node(var["name"], type=var["type"])
        
        # Add edges (connections)
        connections = parsed_data.get("connections", [])
        for conn in connections:
            G.add_edge(conn["source"], conn["target"])
        
        if len(G.nodes()) > 0:
            # Create network plot
            plt.figure(figsize=(12, 10))
            
            # Use spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, 
                                 node_color='lightblue',
                                 node_size=1000,
                                 alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, 
                                 edge_color='gray',
                                 arrows=True,
                                 arrowsize=20,
                                 alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title(f"{model_name} - Network Graph")
            plt.axis('off')
            plt.tight_layout()
            
            plot_file = output_dir / f"{model_name}_network_graph.png"
            _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations.append(str(plot_file))
        
    except Exception as e:
        print(f"Error generating network visualizations: {e}")
    
    return visualizations

def generate_combined_analysis(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate combined analysis visualizations.
    
    Args:
        parsed_data: Parsed GNN data
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        List of generated visualization file paths
    """
    visualizations = []
    
    if not MATPLOTLIB_AVAILABLE:
        return visualizations
    
    try:
        # Create combined analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Variable type distribution
        variables = parsed_data.get("variables", [])
        if variables:
            var_types = [v["type"] for v in variables]
            type_counts = {}
            for var_type in var_types:
                type_counts[var_type] = type_counts.get(var_type, 0) + 1
            
            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            ax1.set_title("Variable Type Distribution")
        
        # 2. Connection count histogram
        connections = parsed_data.get("connections", [])
        if connections:
            source_counts = {}
            target_counts = {}
            for conn in connections:
                source_counts[conn["source"]] = source_counts.get(conn["source"], 0) + 1
                target_counts[conn["target"]] = target_counts.get(conn["target"], 0) + 1
            
            all_nodes = set(source_counts.keys()) | set(target_counts.keys())
            node_counts = [source_counts.get(node, 0) + target_counts.get(node, 0) for node in all_nodes]
            
            ax2.hist(node_counts, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title("Node Connection Count Distribution")
            ax2.set_xlabel("Number of Connections")
            ax2.set_ylabel("Frequency")
        
        # 3. Matrix statistics
        matrices = parsed_data.get("matrices", [])
        if matrices:
            matrix_sizes = [m["data"].size for m in matrices if m["data"] is not None]
            if matrix_sizes:
                ax3.hist(matrix_sizes, bins=min(10, len(matrix_sizes)), alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_title("Matrix Size Distribution")
                ax3.set_xlabel("Matrix Size (elements)")
                ax3.set_ylabel("Frequency")
        
        # 4. Section content length
        sections = parsed_data.get("sections", {})
        if sections:
            section_lengths = [len(content) for content in sections.values()]
            section_names = list(sections.keys())
            
            ax4.bar(range(len(section_names)), section_lengths, alpha=0.7, color='orange')
            ax4.set_title("Section Content Length")
            ax4.set_xlabel("Sections")
            ax4.set_ylabel("Number of Lines")
            ax4.set_xticks(range(len(section_names)))
            ax4.set_xticklabels(section_names, rotation=45, ha='right')
        
        plt.suptitle(f"{model_name} - Combined Analysis", fontsize=16)
        plt.tight_layout()
        
        plot_file = output_dir / f"{model_name}_combined_analysis.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations.append(str(plot_file))
        
    except Exception as e:
        print(f"Error generating combined analysis: {e}")
    
    return visualizations

def generate_combined_visualizations(gnn_files: List[Path], results_dir: Path, verbose: bool = False) -> List[str]:
    """
    Generate combined visualizations across multiple GNN files.
    
    Args:
        gnn_files: List of GNN file paths
        results_dir: Results directory
        verbose: Enable verbose output
        
    Returns:
        List of generated visualization file paths
    """
    visualizations = []
    
    if not MATPLOTLIB_AVAILABLE:
        return visualizations
    
    try:
        # Collect data from all files
        all_variables = []
        all_connections = []
        all_matrices = []
        
        for gnn_file in gnn_files:
            with open(gnn_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parsed_data = parse_gnn_content(content)
            all_variables.extend(parsed_data.get("variables", []))
            all_connections.extend(parsed_data.get("connections", []))
            all_matrices.extend(parsed_data.get("matrices", []))
        
        # Create combined analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall variable type distribution
        if all_variables:
            var_types = [v["type"] for v in all_variables]
            type_counts = {}
            for var_type in var_types:
                type_counts[var_type] = type_counts.get(var_type, 0) + 1
            
            ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            ax1.set_title("Overall Variable Type Distribution")
        
        # 2. File comparison
        file_stats = []
        for gnn_file in gnn_files:
            with open(gnn_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parsed_data = parse_gnn_content(content)
            file_stats.append({
                "name": gnn_file.stem,
                "variables": len(parsed_data.get("variables", [])),
                "connections": len(parsed_data.get("connections", [])),
                "matrices": len(parsed_data.get("matrices", []))
            })
        
        if file_stats:
            file_names = [stat["name"] for stat in file_stats]
            var_counts = [stat["variables"] for stat in file_stats]
            conn_counts = [stat["connections"] for stat in file_stats]
            
            x = range(len(file_names))
            width = 0.35
            
            ax2.bar([i - width/2 for i in x], var_counts, width, label='Variables', alpha=0.7)
            ax2.bar([i + width/2 for i in x], conn_counts, width, label='Connections', alpha=0.7)
            ax2.set_title("File Comparison")
            ax2.set_xlabel("Files")
            ax2.set_ylabel("Count")
            ax2.set_xticks(x)
            ax2.set_xticklabels(file_names, rotation=45, ha='right')
            ax2.legend()
        
        # 3. Matrix size distribution
        if all_matrices:
            matrix_sizes = [m["data"].size for m in all_matrices if m["data"] is not None]
            if matrix_sizes:
                ax3.hist(matrix_sizes, bins=min(15, len(matrix_sizes)), alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_title("Overall Matrix Size Distribution")
                ax3.set_xlabel("Matrix Size (elements)")
                ax3.set_ylabel("Frequency")
        
        # 4. Connection type analysis
        if all_connections:
            connection_types = {}
            for conn in all_connections:
                conn_type = f"{conn['source']}->{conn['target']}"
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
            
            if connection_types:
                top_connections = sorted(connection_types.items(), key=lambda x: x[1], reverse=True)[:10]
                conn_names, conn_counts = zip(*top_connections)
                
                ax4.barh(range(len(conn_names)), conn_counts, alpha=0.7, color='orange')
                ax4.set_title("Top Connection Types")
                ax4.set_xlabel("Count")
                ax4.set_yticks(range(len(conn_names)))
                ax4.set_yticklabels(conn_names)
        
        plt.suptitle("Combined Analysis Across All Files", fontsize=16)
        plt.tight_layout()
        
        plot_file = results_dir / "combined_analysis.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations.append(str(plot_file))
        
    except Exception as e:
        print(f"Error generating combined visualizations: {e}")
    
    return visualizations
