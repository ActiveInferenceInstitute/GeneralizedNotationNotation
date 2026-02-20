#!/usr/bin/env python3
"""
Visualization processor module for GNN Processing Pipeline.

This module provides the main visualization processing functionality.

Functions are organized into sub-modules:
- network_visualizations: Network graph generation, interactive plotly networks, statistics
- combined_analysis: Combined analysis plots, standalone panels, generative model diagrams
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
    def log_step_start(logger, msg): logger.info(f"\U0001f680 {msg}")
    def log_step_success(logger, msg): logger.info(f"\u2705 {msg}")
    def log_step_error(logger, msg): logger.error(f"\u274c {msg}")
    def log_step_warning(logger, msg): logger.warning(f"\u26a0\ufe0f {msg}")
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
    from .network_visualizations import generate_network_visualizations
    from .combined_analysis import generate_combined_analysis, generate_combined_visualizations

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
    from .network_visualizations import generate_network_visualizations
    from .combined_analysis import generate_combined_analysis

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
                elif ('->' in stripped_line or '\u2192' in stripped_line or '>' in stripped_line or ('-' in stripped_line and current_section == 'Connections')):
                    # Connection definition (supports ->, \u2192, > formats, and - in Connections section)
                    if '->' in stripped_line:
                        conn_parts = stripped_line.split('->', 1)
                    elif '\u2192' in stripped_line:
                        conn_parts = stripped_line.split('\u2192', 1)
                    elif '>' in stripped_line:
                        conn_parts = stripped_line.split('>', 1)
                    else:
                        conn_parts = stripped_line.split('-', 1)

                    if len(conn_parts) == 2:
                        source = conn_parts[0].strip()
                        target = conn_parts[1].strip()
                        # Only add if both source and target are non-empty and look like variable names
                        if source and target and (source.replace('_', '').replace('-', '').isalnum() or source in ['s', 'o', '\u03c0', 'u']) and \
                           (target.replace('_', '').replace('-', '').isalnum() or target in ['s', 'o', '\u03c0', 'u']):
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

# --- Re-export everything from sub-modules for backward compatibility ---
from .network_visualizations import (
    generate_network_visualizations,
    _normalize_connection_format,
    _determine_connection_type,
    _get_edge_style,
    _generate_network_statistics,
    _generate_interactive_network,
    _generate_3d_surface_plot,
)

from .combined_analysis import (
    generate_combined_analysis,
    generate_combined_visualizations,
)
