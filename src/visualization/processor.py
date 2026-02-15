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
import warnings
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

# Optional plotly import for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    PLOTLY_AVAILABLE = False

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

# Import analysis utilities
try:
    from analysis.analyzer import parse_matrix_data, generate_matrix_visualizations
except ImportError:
    # Fallback definition if analysis module is not available
    def parse_matrix_data(matrix_str: str) -> Any:
        try:
            import re
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', matrix_str)
            if len(numbers) >= 1:
                return np.array([float(n) for n in numbers])
            return None
        except Exception:
            return None
            
    def generate_matrix_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
        return []

# Set up logger
logger = logging.getLogger(__name__)

# Safe plot saving helper to avoid crashes due to extreme DPI or backend issues
def _save_plot_safely(plot_path: Path, dpi: int = 300, **savefig_kwargs) -> bool:
    """Attempt to save a matplotlib figure with fallback DPI strategies.

    Returns True on success, False on failure.
    """
    def _safe_dpi_value(dpi_input):
        """Validate and sanitize DPI value."""
        try:
            dpi_val = int(dpi_input) if isinstance(dpi_input, (int, float)) else 150
            # Ensure DPI is within reasonable bounds
            return max(50, min(dpi_val, 600))
        except (ValueError, TypeError, OverflowError):
            return 150
    
    # Sanitize DPI value first
    safe_dpi = _safe_dpi_value(dpi)
    
    try:
        plt.savefig(plot_path, dpi=safe_dpi, **savefig_kwargs)
        logger.debug(f"Successfully saved plot with DPI {safe_dpi}")
        return True
    except Exception as e:
        logger.debug(f"Error saving with DPI {safe_dpi}: {e}")
        try:
            fallback_dpi = _safe_dpi_value(matplotlib.rcParams.get('savefig.dpi', 100))
            plt.savefig(plot_path, dpi=fallback_dpi, **savefig_kwargs)
            logger.debug(f"Saved with fallback DPI {fallback_dpi}")
            return True
        except Exception as e2:
            logger.debug(f"Error with fallback DPI: {e2}")
            try:
                # Final fallback - no DPI specified
                plt.savefig(plot_path, **savefig_kwargs)
                logger.debug("Saved with default DPI")
                return True
            except Exception as e3:
                logger.error(f"Failed to save plot {plot_path}: {e3}")
                return False


def _safe_tight_layout():
    """Apply tight_layout with warning suppression.
    
    Tight layout may fail for complex figures - this is not critical.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, 
                                  message='.*[Tt]ight.?layout.*')
            plt.tight_layout()
    except Exception:
        # Tight layout is not critical - silently skip if it fails
        pass

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
        results_dir = output_dir
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
                                        if (any(source in var_names for source in conn.get("source_variables", [])) and
                                            any(target in var_names for target in conn.get("target_variables", [])))]
            
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
        
        # 1. Network visualizations
        if len(parsed_data.get("variables", [])) <= 200:
            try:
                network_viz = generate_network_visualizations(parsed_data, model_dir, model_name)
                visualizations.extend(network_viz)
            except Exception as e:
                if verbose:
                    print(f"Network visualization failed for {model_name}: {e}")
        elif verbose:
            print(f"Skipping network visualizations for {model_name} - too many nodes")
        
        # 2. Matrix visualizations
        try:
            from .matrix_visualizer import MatrixVisualizer
            mv = MatrixVisualizer()
            
            # Try to load from parsed JSON file first (from step 3 GNN processing)
            matrices = {}
            try:
                from pipeline.config import get_output_dir_for_script
                gnn_output_dir = get_output_dir_for_script("3_gnn.py", output_dir.parent if output_dir.name.endswith("_output") else output_dir)
                parsed_json_file = gnn_output_dir / model_name / f"{model_name}_parsed.json"
                
                if parsed_json_file.exists():
                    import json
                    with open(parsed_json_file, 'r') as f:
                        gnn_parsed_data = json.load(f)
                    # Extract from parameters field (correct location)
                    parameters = gnn_parsed_data.get("parameters", [])
                    matrices = mv.extract_matrix_data_from_parameters(parameters)
                    if verbose and matrices:
                        logger.info(f"Extracted {len(matrices)} matrices from parsed JSON for {model_name}")
            except Exception as e:
                if verbose:
                    logger.debug(f"Could not load from parsed JSON: {e}")
            
            # Fallback: Extract from parsed_data (from parse_gnn_content)
            if not matrices:
                # Try parameters field first (correct location)
                parameters = parsed_data.get("parameters", [])
                if parameters:
                    matrices = mv.extract_matrix_data_from_parameters(parameters)
                
                # If still no matrices, try variables (for backward compatibility)
                if not matrices:
                    matrices = mv.extract_matrix_data_from_parameters(parsed_data.get("variables", []))
                
                # Last resort: try matrices section
                if not matrices:
                    for m_info in parsed_data.get("matrices", []):
                        if "data" in m_info:
                            m_name = m_info.get("name", f"matrix_{len(matrices)}")
                            try:
                                import numpy as np
                                m_data = np.array(m_info["data"], dtype=float)
                                matrices[m_name] = m_data
                            except Exception:
                                continue
            
            # Generate visualizations for extracted matrices
            if matrices:
                for m_name, m_data in matrices.items():
                    # Check matrix dimensionality and use appropriate visualization
                    if m_data.ndim == 3:
                        # Use specialized 3D tensor visualization for POMDP transition matrices
                        m_path = model_dir / f"{model_name}_{m_name}_tensor.png"
                        if mv.generate_3d_tensor_visualization(m_name, m_data, m_path, tensor_type="transition"):
                            visualizations.append(str(m_path))
                            # Also generate detailed POMDP analysis
                            analysis_path = model_dir / f"{model_name}_{m_name}_analysis.png"
                            mv.generate_pomdp_transition_analysis(m_data, analysis_path)
                            visualizations.append(str(analysis_path))
                    else:
                        # Use standard 2D heatmap for 1D/2D matrices
                        m_path = model_dir / f"{model_name}_{m_name}_heatmap.png"
                        if mv.generate_matrix_heatmap(m_name, m_data, m_path):
                            visualizations.append(str(m_path))
                if verbose:
                    logger.info(f"Generated {len(matrices)} matrix visualizations for {model_name}")
            else:
                if verbose:
                    logger.warning(f"No matrix data found for {model_name} - checked parameters, variables, and matrices sections")
        except Exception as e:
            if verbose:
                logger.error(f"Matrix visualization failed for {model_name}: {e}")
                import traceback
                traceback.print_exc()

        # 3. Combined analysis
        try:
            analysis_viz = generate_combined_analysis(parsed_data, model_dir, model_name)
            visualizations.extend(analysis_viz)
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
        Dictionary with parsed GNN data including:
        - sections: dict of section name -> list of lines
        - raw_sections: dict of section name -> raw content string (for length analysis)
        - variables: list of variable definitions
        - connections: list of connection definitions
        - matrices: list of matrix definitions
        - parameters: list of parameter definitions from InitialParameterization
        - metadata: dict of metadata
    """
    try:
        parsed = {
            "sections": {},
            "raw_sections": {},  # Store raw section content for length analysis
            "variables": [],
            "connections": [],
            "matrices": [],
            "parameters": [],  # Store parsed parameters from InitialParameterization
            "metadata": {}
        }
        
        lines = content.split('\n')
        current_section = None
        current_param_name = None
        current_param_lines = []
        in_multiline_param = False
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            # Check for section headers (only ## headers, not single # comments)
            # GNN format uses ## for section headers, # for inline comments
            if stripped_line.startswith('##') and not stripped_line.startswith('###'):
                # Save previous section's multiline parameter if any
                if current_param_name and current_param_lines:
                    _save_parameter(parsed, current_param_name, current_param_lines)
                    current_param_name = None
                    current_param_lines = []
                    in_multiline_param = False
                
                current_section = stripped_line.lstrip('#').strip()
                parsed["sections"][current_section] = []
                parsed["raw_sections"][current_section] = ""
            elif current_section:
                parsed["sections"][current_section].append(stripped_line)
                # Accumulate raw content with newlines preserved
                if parsed["raw_sections"][current_section]:
                    parsed["raw_sections"][current_section] += "\n" + line
                else:
                    parsed["raw_sections"][current_section] = line
                
                # Handle InitialParameterization section for parameter extraction
                if current_section == "InitialParameterization":
                    # Check for parameter definition start (e.g., A={, B={, C=)
                    if '=' in stripped_line and not stripped_line.startswith('#'):
                        # Save previous parameter if any
                        if current_param_name and current_param_lines:
                            _save_parameter(parsed, current_param_name, current_param_lines)
                            current_param_lines = []
                        
                        # Extract parameter name
                        eq_pos = stripped_line.find('=')
                        current_param_name = stripped_line[:eq_pos].strip()
                        param_value_part = stripped_line[eq_pos+1:].strip()
                        
                        # Check if it's a single-line or multi-line definition
                        if param_value_part:
                            current_param_lines = [param_value_part]
                            # Check if it's complete (matching braces)
                            if _is_complete_parameter(param_value_part):
                                _save_parameter(parsed, current_param_name, current_param_lines)
                                current_param_name = None
                                current_param_lines = []
                                in_multiline_param = False
                            else:
                                in_multiline_param = True
                    elif in_multiline_param and current_param_name:
                        # Continue multi-line parameter
                        current_param_lines.append(stripped_line)
                        # Check if complete
                        full_value = ' '.join(current_param_lines)
                        if _is_complete_parameter(full_value):
                            _save_parameter(parsed, current_param_name, current_param_lines)
                            current_param_name = None
                            current_param_lines = []
                            in_multiline_param = False
                
                # Extract variables and connections
                # Handle multiple variable definition formats:
                # 1. GNN format: var[dimensions,type=float]
                # 2. Standard format: var: type
                if ':' in stripped_line and '=' not in stripped_line:
                    # Standard format: var: type
                    var_parts = stripped_line.split(':', 1)
                    if len(var_parts) == 2:
                        var_name = var_parts[0].strip()
                        var_type = var_parts[1].strip()
                        parsed["variables"].append({
                            "name": var_name,
                            "type": var_type
                        })
                elif '[' in stripped_line and 'type=' in stripped_line:
                    # GNN format: var[dimensions,type=float]
                    # Extract variable name (everything before [)
                    bracket_pos = stripped_line.find('[')
                    if bracket_pos != -1:
                        var_name = stripped_line[:bracket_pos].strip()
                        # Extract type information
                        type_start = stripped_line.find('type=', bracket_pos)
                        if type_start != -1:
                            type_end = stripped_line.find(',', type_start) if ',' in stripped_line[type_start:] else stripped_line.find(']', type_start)
                            if type_end == -1:
                                type_end = len(stripped_line)
                            var_type = stripped_line[type_start:type_end].strip()
                            parsed["variables"].append({
                                "name": var_name,
                                "type": var_type
                            })
                elif ('->' in stripped_line or '→' in stripped_line or '>' in stripped_line or ('-' in stripped_line and current_section == 'Connections')):
                    # Connection definition (supports ->, →, > formats, and - in Connections section)
                    if '->' in stripped_line:
                        conn_parts = stripped_line.split('->', 1)
                    elif '→' in stripped_line:
                        conn_parts = stripped_line.split('→', 1)
                    elif '>' in stripped_line:
                        conn_parts = stripped_line.split('>', 1)
                    else:
                        conn_parts = stripped_line.split('-', 1)

                    if len(conn_parts) == 2:
                        source = conn_parts[0].strip()
                        target = conn_parts[1].strip()
                        # Only add if both source and target are non-empty and look like variable names
                        if source and target and (source.replace('_', '').replace('-', '').isalnum() or source in ['s', 'o', 'π', 'u']) and \
                           (target.replace('_', '').replace('-', '').isalnum() or target in ['s', 'o', 'π', 'u']):
                            parsed["connections"].append({
                                "source": source,
                                "target": target
                            })
                elif ('{' in stripped_line and '}' in stripped_line) or ('[' in stripped_line and ']' in stripped_line):
                    # Potential matrix definition (supports both tuple and bracket formats)
                    try:
                        matrix_data = parse_matrix_data(stripped_line)
                        if matrix_data is not None:
                            parsed["matrices"].append({
                                "data": matrix_data,
                                "definition": stripped_line
                            })
                    except Exception:
                        pass
        
        # Save any remaining multiline parameter
        if current_param_name and current_param_lines:
            _save_parameter(parsed, current_param_name, current_param_lines)
        
        return parsed
        
    except Exception as e:
        return {
            "error": str(e),
            "sections": {},
            "raw_sections": {},
            "variables": [],
            "connections": [],
            "matrices": [],
            "parameters": [],
            "metadata": {}
        }


def _is_complete_parameter(value_str: str) -> bool:
    """Check if a parameter value string has matching braces/parentheses."""
    open_braces = value_str.count('{')
    close_braces = value_str.count('}')
    open_parens = value_str.count('(')
    close_parens = value_str.count(')')
    return open_braces == close_braces and open_parens == close_parens and (open_braces > 0 or open_parens > 0)


def _save_parameter(parsed: Dict[str, Any], param_name: str, param_lines: List[str]) -> None:
    """Parse and save a parameter definition."""
    try:
        full_value = ' '.join(param_lines)
        # Parse the tuple/matrix format
        parsed_value = _parse_parameter_value(full_value)
        if parsed_value is not None:
            parsed["parameters"].append({
                "name": param_name,
                "value": parsed_value,
                "raw": full_value
            })
    except Exception as e:
        # Still save as raw if parsing fails
        parsed["parameters"].append({
            "name": param_name,
            "value": None,
            "raw": ' '.join(param_lines),
            "parse_error": str(e)
        })


def _parse_parameter_value(value_str: str) -> Any:
    """
    Parse a parameter value string like {(0.9, 0.05, 0.05), ...} into a Python structure.
    """
    try:
        # Clean up the string
        cleaned = value_str.strip()
        
        # Replace tuple notation with list notation for easier parsing
        # Convert {(...), (...)} to [[...], [...]]
        cleaned = cleaned.replace('{', '[').replace('}', ']')
        cleaned = cleaned.replace('(', '[').replace(')', ']')
        
        # Use ast.literal_eval for safe parsing
        import ast
        result = ast.literal_eval(cleaned)
        return result
    except Exception:
        return None

# Matrix parsing and visualization logic moved to analyzer.py

def generate_network_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """
    Generate network visualizations for POMDP models.

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

        # Extract variables from parsed data
        variables = parsed_data.get("variables", [])

        # Ensure variables is a list of dictionaries
        if not isinstance(variables, list):
            variables = []

        # Extract connections from parsed data
        connections = parsed_data.get("connections", [])

        # Ensure connections is a list
        if not isinstance(connections, list):
            connections = []

        # Add nodes (variables) with proper type information
        for var_info in variables:
            if isinstance(var_info, dict):
                var_name = var_info.get("name", "unknown")
                var_type = var_info.get("var_type", "unknown")
                dimensions = var_info.get("dimensions", [])
                description = var_info.get("description", "")

                # Create comprehensive node attributes
                node_attrs = {
                    'type': var_type,
                    'dimensions': dimensions,
                    'description': description,
                    'size': max(1, min(10, len(dimensions) * 2)),  # Node size based on dimensions
                }
                G.add_node(var_name, **node_attrs)

        # Add edges (connections) - handle both old and new connection formats
        for conn_info in connections:
            if isinstance(conn_info, dict):
                # Normalize connection format to handle both old and new formats
                normalized_conn = _normalize_connection_format(conn_info)
                source_vars = normalized_conn.get("source_variables", [])
                target_vars = normalized_conn.get("target_variables", [])

                # Add edges between all source-target pairs
                for source_var in source_vars:
                    for target_var in target_vars:
                        if source_var and target_var and source_var != target_var:
                            # Determine connection type based on variable types
                            source_type = None
                            target_type = None

                            # Find variable types
                            for var in variables:
                                if isinstance(var, dict) and var.get("name") == source_var:
                                    source_type = var.get("var_type", "unknown")
                                if isinstance(var, dict) and var.get("name") == target_var:
                                    target_type = var.get("var_type", "unknown")

                            # Determine connection type based on POMDP semantics
                            conn_type = _determine_connection_type(source_var, target_var, source_type, target_type)

                            # Add edge with comprehensive metadata
                            edge_attrs = {
                                'connection_type': conn_type,
                                'source_location': normalized_conn.get("source_location"),
                                'metadata': normalized_conn.get("metadata", {}),
                                'source_type': source_type,
                                'target_type': target_type,
                                'weight': 1.0,  # Default weight
                                'style': _get_edge_style(conn_type)
                            }
                            G.add_edge(source_var, target_var, **edge_attrs)

        if len(G.nodes()) > 0:
            # Create network plot
            plt.figure(figsize=(14, 12))

            # Use spring layout with better parameters for readability
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

            # Get node attributes for coloring and sizing
            node_sizes = [G.nodes[node].get('size', 5) * 100 for node in G.nodes()]
            node_types = [G.nodes[node].get('type', 'unknown') for node in G.nodes()]

            # Define color mapping for different variable types
            type_colors = {
                'hidden_state': 'skyblue',
                'observation': 'lightgreen',
                'policy': 'lightcoral',
                'action': 'gold',
                'prior_vector': 'plum',
                'likelihood_matrix': 'orange',
                'transition_matrix': 'pink',
                'preference_vector': 'lightblue',
                'unknown': 'gray'
            }

            node_colors = [type_colors.get(node_type, 'gray') for node_type in node_types]

            # Draw the network with connection-type-specific styling
            # Group edges by connection type for different styling
            edge_groups = {}
            for edge in G.edges(data=True):
                conn_type = edge[2].get('connection_type', 'generic_causal')
                if conn_type not in edge_groups:
                    edge_groups[conn_type] = []
                edge_groups[conn_type].append((edge[0], edge[1]))

            # Draw edges by type
            for conn_type, edges in edge_groups.items():
                style = _get_edge_style(conn_type)
                edge_list = [(u, v) for u, v in edges]

                if edge_list:
                    nx.draw_networkx_edges(G, pos,
                                         edgelist=edge_list,
                                         edge_color=style['color'],
                                         width=style['width'],
                                         alpha=style['alpha'],
                                         arrows=True,
                                         arrowsize=20,
                                         style=style['style'])

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, fc=color, label=var_type)
                              for var_type, color in type_colors.items()]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

            plt.title(f'Bayesian Graphical Model: {model_name}\nPOMDP Active Inference Network', fontsize=16, fontweight='bold')
            plt.axis('off')
            _safe_tight_layout()

            # Save network visualization
            network_path = output_dir / f"{model_name}_network_graph.png"
            plt.savefig(network_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(network_path))

            # Generate network statistics
            stats = _generate_network_statistics(variables, connections)
            stats_path = output_dir / f"{model_name}_network_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            visualizations.append(str(stats_path))

            # Generate interactive network if plotly available
            if 'plotly' in globals() and plotly:
                try:
                    interactive_path = _generate_interactive_network(G, output_dir / f"{model_name}_network_interactive.html")
                    if interactive_path:
                        visualizations.append(str(interactive_path))
                except Exception as e:
                    print(f"Failed to generate interactive network: {e}")

        else:
            print(f"Warning: No valid nodes found for network visualization of {model_name}")

    except Exception as e:
        print(f"Error generating network visualizations for {model_name}: {e}")
        import traceback
        traceback.print_exc()

    return visualizations

def _normalize_connection_format(conn_info: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize connection format to handle both old and new formats."""
    if "source_variables" in conn_info and "target_variables" in conn_info:
        # New format with arrays
        return conn_info
    elif "source" in conn_info and "target" in conn_info:
        # Old format with single values
        return {
            "source_variables": [conn_info["source"]],
            "target_variables": [conn_info["target"]],
            **{k: v for k, v in conn_info.items() if k not in ["source", "target"]}
        }
    else:
        # Unknown format - return as-is
        return conn_info


def _determine_connection_type(source_var: str, target_var: str, source_type: str = None, target_type: str = None) -> str:
    """Determine the semantic type of connection between variables."""
    # POMDP-specific connection types
    if source_type and target_type:
        # State to state (with action)
        if source_type == "hidden_state" and target_type == "hidden_state":
            return "state_transition"
        # State to observation
        elif source_type == "hidden_state" and target_type == "observation":
            return "observation_generation"
        # State to transition matrix
        elif source_type == "hidden_state" and "transition" in target_type:
            return "state_action_influence"
        # Action to state
        elif source_type == "action" and target_type == "hidden_state":
            return "action_effect"
        # Policy to action
        elif source_type == "policy" and target_type == "action":
            return "policy_selection"
        # Prior to state
        elif source_type == "prior_vector" and target_type == "hidden_state":
            return "prior_influence"
        # Likelihood matrix connections
        elif source_type == "hidden_state" and "likelihood" in target_type:
            return "likelihood_influence"
        # Free energy connections
        elif "free_energy" in source_type or "free_energy" in target_type:
            return "energy_flow"

    # Generic semantic types based on variable names
    if source_var == "s" and target_var in ["A", "o"]:
        return "state_observation"
    elif source_var in ["s", "s_prime"] and target_var == "B":
        return "state_transition_matrix"
    elif source_var == "C" and target_var == "G":
        return "preference_energy"
    elif source_var == "E" and target_var == "π":
        return "habit_policy"
    elif source_var == "π" and target_var == "u":
        return "policy_action"

    return "generic_causal"


def _get_edge_style(connection_type: str) -> Dict[str, Any]:
    """Get visual styling for different connection types."""
    style_map = {
        "state_transition": {"color": "blue", "width": 3, "alpha": 0.8, "style": "solid"},
        "observation_generation": {"color": "green", "width": 2, "alpha": 0.7, "style": "dashed"},
        "state_action_influence": {"color": "orange", "width": 2, "alpha": 0.7, "style": "dotted"},
        "action_effect": {"color": "red", "width": 3, "alpha": 0.8, "style": "solid"},
        "policy_selection": {"color": "purple", "width": 2, "alpha": 0.7, "style": "solid"},
        "prior_influence": {"color": "cyan", "width": 2, "alpha": 0.6, "style": "dashed"},
        "likelihood_influence": {"color": "magenta", "width": 2, "alpha": 0.6, "style": "dotted"},
        "energy_flow": {"color": "yellow", "width": 1, "alpha": 0.5, "style": "dashed"},
        "preference_energy": {"color": "lime", "width": 2, "alpha": 0.7, "style": "solid"},
        "habit_policy": {"color": "pink", "width": 2, "alpha": 0.7, "style": "solid"},
        "generic_causal": {"color": "gray", "width": 1, "alpha": 0.5, "style": "solid"}
    }

    return style_map.get(connection_type, style_map["generic_causal"])

def _generate_network_statistics(variables: list, connections: list) -> Dict[str, Any]:
    """Generate comprehensive network statistics."""
    stats = {
        "total_variables": len(variables),
        "total_connections": len(connections),
        "variable_types": {},
        "connection_types": {},
        "network_properties": {}
    }

    # Count variable types
    for var_info in variables:
        if isinstance(var_info, dict):
            var_type = var_info.get("var_type", "unknown")
            stats["variable_types"][var_type] = stats["variable_types"].get(var_type, 0) + 1

    # Count connection types (based on variable relationships)
    for conn_info in connections:
        if isinstance(conn_info, dict):
            # Normalize connection format
            normalized_conn = _normalize_connection_format(conn_info)
            source_vars = normalized_conn.get("source_variables", [])
            target_vars = normalized_conn.get("target_variables", [])

            for source_var in source_vars:
                for target_var in target_vars:
                    if source_var != target_var:
                        # Determine connection type based on variable types
                        conn_type = f"{source_var}->{target_var}"
                        stats["connection_types"][conn_type] = stats["connection_types"].get(conn_type, 0) + 1

    # Network properties (if networkx available)
    if NETWORKX_AVAILABLE:
        try:
            # Create a simple graph for network analysis
            simple_G = nx.DiGraph()
            for conn_info in connections:
                # Normalize connection format for graph creation
                normalized_conn = _normalize_connection_format(conn_info)
                source_vars = normalized_conn.get("source_variables", [])
                target_vars = normalized_conn.get("target_variables", [])
                for source_var in source_vars:
                    for target_var in target_vars:
                        if source_var != target_var:
                            simple_G.add_edge(source_var, target_var)

            if len(simple_G.nodes()) > 0:
                stats["network_properties"] = {
                    "num_nodes": simple_G.number_of_nodes(),
                    "num_edges": simple_G.number_of_edges(),
                    "density": nx.density(simple_G),
                    "is_strongly_connected": nx.is_strongly_connected(simple_G) if len(simple_G.nodes()) > 1 else True,
                    "is_weakly_connected": nx.is_weakly_connected(simple_G) if len(simple_G.nodes()) > 1 else True,
                    "average_clustering": nx.average_clustering(simple_G)
                }
        except Exception as e:
            print(f"Error calculating network properties: {e}")

    return stats

def _generate_interactive_network(G, output_path: Path) -> bool:
    """Generate interactive network visualization using plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Extract node positions and attributes
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create node traces
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []

        type_colors = {
            'hidden_state': 'skyblue',
            'observation': 'lightgreen',
            'policy': 'lightcoral',
            'action': 'gold',
            'prior_vector': 'plum',
            'likelihood_matrix': 'orange',
            'transition_matrix': 'pink',
            'preference_vector': 'lightblue',
            'unknown': 'gray'
        }

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            node_type = G.nodes[node].get('type', 'unknown')
            node_colors.append(type_colors.get(node_type, 'gray'))

            node_size = G.nodes[node].get('size', 5) * 10
            node_sizes.append(node_size)

            # Create hover text
            hover_text = f"<b>{node}</b><br>Type: {node_type}"
            if 'dimensions' in G.nodes[node]:
                hover_text += f"<br>Dimensions: {G.nodes[node]['dimensions']}"
            if 'description' in G.nodes[node]:
                hover_text += f"<br>Description: {G.nodes[node]['description'][:100]}..."
            node_text.append(hover_text)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black'),
                sizemode='diameter'
            ),
            hovertext=node_text,
            name='Variables'
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title=f'Interactive Bayesian Graphical Model: {output_path.stem}',
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text="Variable Types",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=0.995,
                               xanchor='left', yanchor='top'
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))

        # Save interactive HTML
        fig.write_html(str(output_path))
        return True

    except Exception as e:
        print(f"Error generating interactive network: {e}")
        return False

def _generate_3d_surface_plot(matrix: np.ndarray, matrix_name: str, output_path: Path) -> bool:
    """Generate a 3D surface plot for a matrix."""
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        return False

    try:
        # Sample large matrices for 3D visualization
        if matrix.size > 1000:
            # Use every 5th point for large matrices
            step = max(1, matrix.shape[0] // 20)
            x = np.arange(0, matrix.shape[1], step)
            y = np.arange(0, matrix.shape[0], step)
            X, Y = np.meshgrid(x, y)
            Z = matrix[::step, ::step]
        else:
            x = np.arange(matrix.shape[1])
            y = np.arange(matrix.shape[0])
            X, Y = np.meshgrid(x, y)
            Z = matrix

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        ax.set_title(f'3D Surface Plot: {matrix_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Value')

        _safe_tight_layout()
        _save_plot_safely(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        print(f"Error generating 3D surface plot for {matrix_name}: {e}")
        plt.close()
        return False

def _generate_interactive_network(G, output_path: Path) -> bool:
    """Generate an interactive network visualization using plotly."""
    if not PLOTLY_AVAILABLE or not go:
        return False

    try:
        # Get node positions using spring layout
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Extract node and edge data
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node traces
        node_x = []
        node_y = []
        node_info = []
        node_types = nx.get_node_attributes(G, 'type')

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_info.append(f'{node}<br>Type: {node_types.get(node, "unknown")}<br>Connections: {G.degree(node)}')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            hovertext=node_info,
            marker=dict(
                size=[G.degree(node) * 10 + 20 for node in G.nodes()],
                color=['lightblue' if node_types.get(node, 'unknown') == 'unknown' else 'orange' for node in G.nodes()],
                line=dict(width=2)
            )
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Interactive Network Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)',
                           title_font_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text="Hover over nodes and edges for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )

        # Save as HTML
        fig.write_html(str(output_path))

        return True

    except Exception as e:
        print(f"Error generating interactive network: {e}")
        return False

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
            var_types = []
            for var_info in variables:
                # Extract type from variable info - could be in different fields
                var_type = 'unknown'
                if isinstance(var_info, dict):
                    var_type = var_info.get('type', var_info.get('node_type', 'unknown'))
                var_types.append(var_type)

            if var_types:
                type_counts = {}
                for var_type in var_types:
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1

                ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
                ax1.set_title("Variable Type Distribution")
            else:
                ax1.text(0.5, 0.5, "No variable type data", transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title("Variable Type Distribution (No Data)")
        
        # 2. Connection count histogram
        connections = parsed_data.get("connections", [])
        if connections:
            source_counts = {}
            target_counts = {}
            for conn in connections:
                if isinstance(conn, dict):
                    source = conn.get("source_variables", [conn.get("source", "unknown")])[0] if conn.get("source_variables") else conn.get("source", "unknown")
                    target = conn.get("target_variables", [conn.get("target", "unknown")])[0] if conn.get("target_variables") else conn.get("target", "unknown")

                    source_counts[source] = source_counts.get(source, 0) + 1
                    target_counts[target] = target_counts.get(target, 0) + 1

            all_nodes = set(source_counts.keys()) | set(target_counts.keys())
            node_counts = [source_counts.get(node, 0) + target_counts.get(node, 0) for node in all_nodes]

            if node_counts:
                ax2.hist(node_counts, bins=min(10, len(set(node_counts))), alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title("Node Connection Count Distribution")
                ax2.set_xlabel("Number of Connections")
                ax2.set_ylabel("Frequency")
            else:
                ax2.text(0.5, 0.5, "No connection data", transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title("Node Connection Count Distribution (No Data)")
        
        # 3. Matrix statistics
        parameters = parsed_data.get("parameters", [])
        matrix_sizes = []
        if parameters:
            for param in parameters:
                if isinstance(param, dict) and "value" in param:
                    value = param["value"]
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        try:
                            import numpy as np
                            arr = np.array(value)
                            matrix_sizes.append(arr.size)
                        except Exception:
                            # Count elements in nested structure
                            def count_elements(obj):
                                if isinstance(obj, (int, float)):
                                    return 1
                                elif isinstance(obj, (list, tuple)):
                                    return sum(count_elements(item) for item in obj)
                                else:
                                    return 1
                            matrix_sizes.append(count_elements(value))

        if matrix_sizes:
            ax3.hist(matrix_sizes, bins=min(10, len(matrix_sizes)), alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title("Matrix Size Distribution")
            ax3.set_xlabel("Matrix Size (elements)")
            ax3.set_ylabel("Frequency")
        else:
            ax3.text(0.5, 0.5, "No matrix data", transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title("Matrix Size Distribution (No Data)")
        
        # 4. Section content length
        raw_sections = parsed_data.get("raw_sections", {})
        if raw_sections:
            section_lengths = [len(str(content)) for content in raw_sections.values()]
            section_names = list(raw_sections.keys())

            ax4.bar(range(len(section_names)), section_lengths, alpha=0.7, color='orange')
            ax4.set_title("Section Content Length")
            ax4.set_xlabel("Sections")
            ax4.set_ylabel("Content Length (characters)")
            ax4.set_xticks(range(len(section_names)))
            ax4.set_xticklabels([name[:20] + "..." if len(name) > 20 else name for name in section_names], rotation=45, ha='right')
        else:
            ax4.text(0.5, 0.5, "No section data", transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title("Section Content Length (No Data)")
        
        plt.suptitle(f"{model_name} - Combined Analysis", fontsize=16)
        _safe_tight_layout()
        
        plot_file = output_dir / f"{model_name}_combined_analysis.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations.append(str(plot_file))
        
        # Generate standalone panel visualizations
        standalone_files = _generate_standalone_panels(parsed_data, output_dir, model_name)
        visualizations.extend(standalone_files)
        
        # Generate generative model diagram
        gm_files = _generate_generative_model_diagram(parsed_data, output_dir, model_name)
        visualizations.extend(gm_files)
        
    except Exception as e:
        print(f"Error generating combined analysis: {e}")
    
    return visualizations


def _generate_standalone_panels(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """Generate standalone visualization files for each panel in the combined analysis."""
    visualizations = []
    
    if not MATPLOTLIB_AVAILABLE:
        return visualizations
    
    try:
        # 1. Matrix Size Distribution (standalone)
        parameters = parsed_data.get("parameters", [])
        matrix_sizes = []
        matrix_names = []
        if parameters:
            for param in parameters:
                if isinstance(param, dict) and "value" in param:
                    value = param["value"]
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        try:
                            arr = np.array(value)
                            matrix_sizes.append(arr.size)
                            matrix_names.append(param.get("name", "Unknown"))
                        except Exception:
                            pass
        
        if matrix_sizes:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(matrix_names, matrix_sizes, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.set_title(f"{model_name} - Matrix Size Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("Matrix Parameter")
            ax.set_ylabel("Size (elements)")
            
            # Add value labels on bars
            for bar, size in zip(bars, matrix_sizes):
                height = bar.get_height()
                ax.annotate(f'{size}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
            
            _safe_tight_layout()
            plot_file = output_dir / f"{model_name}_matrix_size_distribution.png"
            _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(plot_file))
        
        # 2. Section Content Length (standalone)
        raw_sections = parsed_data.get("raw_sections", {})
        if raw_sections:
            fig, ax = plt.subplots(figsize=(12, 6))
            section_lengths = [len(str(content)) for content in raw_sections.values()]
            section_names = list(raw_sections.keys())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(section_names)))
            bars = ax.bar(range(len(section_names)), section_lengths, alpha=0.7, color=colors, edgecolor='black')
            ax.set_title(f"{model_name} - Section Content Length", fontsize=14, fontweight='bold')
            ax.set_xlabel("Sections")
            ax.set_ylabel("Content Length (characters)")
            ax.set_xticks(range(len(section_names)))
            ax.set_xticklabels([name[:15] + "..." if len(name) > 15 else name for name in section_names], 
                               rotation=45, ha='right', fontsize=9)
            
            _safe_tight_layout()
            plot_file = output_dir / f"{model_name}_section_content_length.png"
            _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(plot_file))
        
        # 3. Variable Type Distribution (standalone pie chart)
        variables = parsed_data.get("variables", [])
        if variables:
            fig, ax = plt.subplots(figsize=(10, 8))
            var_types = []
            for var_info in variables:
                var_type = 'unknown'
                if isinstance(var_info, dict):
                    var_type = var_info.get('type', var_info.get('node_type', 'unknown'))
                var_types.append(var_type)
            
            if var_types:
                type_counts = {}
                for var_type in var_types:
                    type_counts[var_type] = type_counts.get(var_type, 0) + 1
                
                # Use a vibrant color palette
                colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
                wedges, texts, autotexts = ax.pie(type_counts.values(), labels=type_counts.keys(), 
                                                   autopct='%1.1f%%', colors=colors)
                ax.set_title(f"{model_name} - Variable Type Distribution", fontsize=14, fontweight='bold')
                
                _safe_tight_layout()
                plot_file = output_dir / f"{model_name}_variable_type_distribution.png"
                _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(plot_file))
        
    except Exception as e:
        print(f"Error generating standalone panels: {e}")
    
    return visualizations


def _generate_generative_model_diagram(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """Generate a generative model diagram showing the POMDP structure."""
    visualizations = []
    
    if not MATPLOTLIB_AVAILABLE:
        return visualizations
    
    try:
        # Create a clean generative model diagram
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Define node positions for standard POMDP
        positions = {
            'D': (2, 8),      # Prior
            's': (2, 6),      # Hidden state (t)
            's\'': (5, 6),    # Hidden state (t+1)
            'A': (2, 4),      # Likelihood
            'o': (2, 2),      # Observation
            'B': (3.5, 7.5),  # Transition
            'C': (8, 4),      # Preferences
            'E': (8, 7),      # Habit
            'π': (6, 5),      # Policy
            'G': (8, 5.5),    # Expected Free Energy
            'u': (5, 3),      # Action
        }
        
        # Node colors by type
        node_colors = {
            'D': '#98D8C8',    # Prior - teal
            's': '#7EC8E3',    # Hidden state - blue
            's\'': '#7EC8E3',  # Hidden state - blue
            'A': '#F7DC6F',    # Likelihood - yellow
            'o': '#82E0AA',    # Observation - green
            'B': '#F1948A',    # Transition - pink
            'C': '#C39BD3',    # Preferences - purple
            'E': '#F5B7B1',    # Habit - light pink
            'π': '#FAD7A0',    # Policy - orange
            'G': '#D2B4DE',    # EFE - lavender
            'u': '#ABEBC6',    # Action - light green
        }
        
        # Draw nodes
        for node_name, (x, y) in positions.items():
            color = node_colors.get(node_name, 'lightgray')
            circle = plt.Circle((x, y), 0.4, color=color, ec='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, node_name, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)
        
        # Define edges based on parsed connections or standard POMDP
        edges = [
            ('D', 's', 'Prior'),
            ('s', 'A', 'Likelihood'),
            ('A', 'o', 'Observation'),
            ('s', 's\'', 'Transition'),
            ('B', 's\'', 'Control'),
            ('π', 'u', 'Selection'),
            ('C', 'G', 'Preferences'),
            ('E', 'π', 'Habit'),
            ('G', 'π', 'EFE'),
            ('u', 's\'', 'Effect'),
        ]
        
        # Draw edges
        for source, target, label in edges:
            if source in positions and target in positions:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                
                # Calculate arrow position (from edge of source to edge of target)
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    # Start from edge of source node
                    start_x = x1 + 0.4 * dx / dist
                    start_y = y1 + 0.4 * dy / dist
                    # End at edge of target node
                    end_x = x2 - 0.4 * dx / dist
                    end_y = y2 - 0.4 * dy / dist
                    
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                               zorder=1)
        
        # Add title and legend
        ax.set_title(f"{model_name}\nGenerative Model Structure (POMDP)", fontsize=16, fontweight='bold', pad=20)
        
        # Add legend with node types
        legend_items = [
            ('D', 'Prior (Initial State Belief)', '#98D8C8'),
            ('s/s\'', 'Hidden State', '#7EC8E3'),
            ('A', 'Likelihood Matrix', '#F7DC6F'),
            ('o', 'Observation', '#82E0AA'),
            ('B', 'Transition Matrix', '#F1948A'),
            ('C', 'Preferences', '#C39BD3'),
            ('E', 'Habit (Policy Prior)', '#F5B7B1'),
            ('π', 'Policy', '#FAD7A0'),
            ('G', 'Expected Free Energy', '#D2B4DE'),
            ('u', 'Action', '#ABEBC6'),
        ]
        
        legend_y = 1.5
        for symbol, desc, color in legend_items:
            ax.add_patch(plt.Rectangle((6.5, legend_y), 0.3, 0.3, color=color, ec='black'))
            ax.text(6.95, legend_y + 0.15, f"{symbol}: {desc}", fontsize=8, va='center')
            legend_y -= 0.4
        
        _safe_tight_layout()
        plot_file = output_dir / f"{model_name}_generative_model.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations.append(str(plot_file))
        
    except Exception as e:
        print(f"Error generating generative model diagram: {e}")
        import traceback
        traceback.print_exc()
    
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
        # Import the proper GNN parser
        from .parser import GNNParser
        
        # Collect data from all files
        all_variables = []
        all_connections = []
        all_matrices = []
        
        parser = GNNParser()
        for gnn_file in gnn_files:
            try:
                parsed_data = parser.parse_file(str(gnn_file))
                
                # Convert Variables dict to list format for consistency
                variables = parsed_data.get("Variables", {})
                for var_name, var_info in variables.items():
                    all_variables.append({
                        "name": var_name,
                        "type": var_info.get('type', 'unknown'),
                        "dimensions": var_info.get('dimensions', []),
                        "comment": var_info.get('comment', '')
                    })
                
                # Add connections
                connections = parsed_data.get("Edges", [])
                all_connections.extend(connections)
                
                # For matrices, create from variables with 2+ dimensions
                for var_name, var_info in variables.items():
                    dimensions = var_info.get('dimensions', [])
                    if len(dimensions) >= 2 and all(isinstance(d, int) for d in dimensions[:2]):
                        all_matrices.append({
                            "name": var_name,
                            "shape": dimensions,
                            "size": dimensions[0] * dimensions[1] if len(dimensions) >= 2 else 0
                        })
                        
            except Exception as e:
                logger.warning(f"Could not parse {gnn_file}: {e}")
        
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
            matrix_sizes = []
            for m in all_matrices:
                if isinstance(m, dict) and "size" in m:
                    matrix_sizes.append(m["size"])
                elif hasattr(m, 'size'):
                    matrix_sizes.append(m.size)
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
        _safe_tight_layout()
        
        plot_file = results_dir / "combined_analysis.png"
        _save_plot_safely(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations.append(str(plot_file))
        
    except Exception as e:
        print(f"Error generating combined visualizations: {e}")
    
    return visualizations
