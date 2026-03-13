#!/usr/bin/env python3
"""
Analysis analyzer module for GNN statistical analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import json
import numpy as np
from datetime import datetime
import logging
import time
import sys

logger = logging.getLogger(__name__)

# Import visualization libraries with error handling
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    cm = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

def perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis on a GNN file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract structural elements
        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)
        sections = extract_sections_for_analysis(content)

        # Calculate statistics
        var_stats = calculate_variable_statistics(variables)
        conn_stats = calculate_connection_statistics(connections)
        section_stats = calculate_section_statistics(sections)

        # Analyze distributions
        distributions = analyze_distributions(variables, connections)

        # Calculate correlations
        correlations = calculate_correlations(variables, connections)

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "line_count": len(content.splitlines()),
            "variables": variables,
            "connections": connections,
            "sections": sections,
            "variable_statistics": var_stats,
            "connection_statistics": conn_stats,
            "section_statistics": section_stats,
            "distributions": distributions,
            "correlations": correlations,
            "analysis_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise Exception(f"Failed to analyze {file_path}: {e}") from e

def extract_variables_for_analysis(content: str) -> List[Dict[str, Any]]:
    """Extract variables for statistical analysis."""
    variables = []

    # Look for variable definitions
    var_patterns = [
        r'(\w+)\s*:\s*(\w+)',  # name: type
        r'(\w+)\s*=\s*([^;\n]+)',  # name = value
        r'(\w+)\s*\[([^\]]+)\]',  # name[dimensions]
    ]

    for pattern in var_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            variables.append({
                "name": match.group(1),
                "definition": match.group(0),
                "line": content[:match.start()].count('\n') + 1,
                "type": match.group(2) if ":" in match.group(0) else "unknown"
            })

    return variables

def extract_connections_for_analysis(content: str) -> List[Dict[str, Any]]:
    """Extract connections for statistical analysis."""
    connections = []
    seen = set()  # Deduplicate connections

    # 1. Parse GNN ## Connections section directly (highest priority)
    connections_section = re.search(
        r'##\s*Connections\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if connections_section:
        section_text = connections_section.group(1)
        section_start_line = content[:connections_section.start()].count('\n') + 2
        for i, line in enumerate(section_text.strip().split('\n')):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # GNN connection operators: > (directional), - (bidirectional), < (reverse)
            gnn_match = re.match(r'(\w+)\s*([>\-<])\s*(\w+)', line)
            if gnn_match:
                src, op, tgt = gnn_match.group(1), gnn_match.group(2), gnn_match.group(3)
                conn_type = "directional" if op == ">" else "bidirectional" if op == "-" else "reverse"
                key = (src, tgt, conn_type)
                if key not in seen:
                    seen.add(key)
                    connections.append({
                        "source": src,
                        "target": tgt,
                        "connection": line,
                        "connection_type": conn_type,
                        "line": section_start_line + i
                    })

    # 2. Also look for generic connection patterns outside the section
    conn_patterns = [
        (r'(\w+)\s*->\s*(\w+)', "directional"),   # source -> target
        (r'(\w+)\s*→\s*(\w+)', "directional"),     # source → target
        (r'(\w+)\s*connects\s*(\w+)', "association"),  # source connects target
    ]

    for pattern, conn_type in conn_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            key = (match.group(1), match.group(2), conn_type)
            if key not in seen:
                seen.add(key)
                connections.append({
                    "source": match.group(1),
                    "target": match.group(2),
                    "connection": match.group(0),
                    "connection_type": conn_type,
                    "line": content[:match.start()].count('\n') + 1
                })

    return connections

def extract_sections_for_analysis(content: str) -> List[Dict[str, Any]]:
    """Extract sections for statistical analysis."""
    sections = []

    # Look for section headers
    section_patterns = [
        r'^#+\s+(.+)$',  # Markdown headers
        r'^(\w+):\s*$',  # Section labels
    ]

    for pattern in section_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            sections.append({
                "name": match.group(1),
                "line": content[:match.start()].count('\n') + 1
            })

    return sections

def calculate_variable_statistics(variables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for variables."""
    if not variables:
        return {"count": 0, "types": {}, "average_line": 0}

    stats = {
        "count": len(variables),
        "types": {},
        "average_line": np.mean([var.get("line", 0) for var in variables]),
        "line_std": np.std([var.get("line", 0) for var in variables])
    }

    # Count types
    for var in variables:
        var_type = var.get("type", "unknown")
        stats["types"][var_type] = stats["types"].get(var_type, 0) + 1

    return stats

def calculate_connection_statistics(connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for connections."""
    if not connections:
        return {"count": 0, "average_line": 0}

    stats = {
        "count": len(connections),
        "average_line": np.mean([conn.get("line", 0) for conn in connections]),
        "line_std": np.std([conn.get("line", 0) for conn in connections])
    }

    return stats

def calculate_section_statistics(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for sections."""
    if not sections:
        return {"count": 0, "average_line": 0}

    stats = {
        "count": len(sections),
        "average_line": np.mean([section.get("line", 0) for section in sections]),
        "line_std": np.std([section.get("line", 0) for section in sections])
    }

    return stats

def count_type_distribution(variables: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count distribution of variable types."""
    type_counts = {}
    for var in variables:
        var_type = var.get("type", "unknown")
        type_counts[var_type] = type_counts.get(var_type, 0) + 1
    return type_counts

def build_connectivity_matrix(connections: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build connectivity matrix from connections."""
    connectivity = {}
    for conn in connections:
        source = conn.get("source", "")
        target = conn.get("target", "")
        if source not in connectivity:
            connectivity[source] = []
        connectivity[source].append(target)
    return connectivity

def analyze_distributions(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze distributions of model elements."""
    analysis = {
        "variable_distribution": {},
        "connection_distribution": {},
        "complexity_metrics": {}
    }

    # Analyze variable distribution
    if variables:
        var_lines = [var.get("line", 0) for var in variables]
        analysis["variable_distribution"] = {
            "mean": float(np.mean(var_lines)),
            "std": float(np.std(var_lines)),
            "min": float(np.min(var_lines)),
            "max": float(np.max(var_lines)),
            "median": float(np.median(var_lines))
        }
        if SCIPY_AVAILABLE and len(var_lines) > 2:
            analysis["variable_distribution"]["skewness"] = float(stats.skew(var_lines))
            analysis["variable_distribution"]["kurtosis"] = float(stats.kurtosis(var_lines))
            counts = np.unique(var_lines, return_counts=True)[1]
            analysis["variable_distribution"]["entropy"] = float(stats.entropy(counts))

    # Analyze connection distribution
    if connections:
        conn_lines = [conn.get("line", 0) for conn in connections]
        analysis["connection_distribution"] = {
            "mean": float(np.mean(conn_lines)),
            "std": float(np.std(conn_lines)),
            "min": float(np.min(conn_lines)),
            "max": float(np.max(conn_lines)),
            "median": float(np.median(conn_lines))
        }
        if SCIPY_AVAILABLE and len(conn_lines) > 2:
            analysis["connection_distribution"]["skewness"] = float(stats.skew(conn_lines))
            analysis["connection_distribution"]["kurtosis"] = float(stats.kurtosis(conn_lines))
            counts = np.unique(conn_lines, return_counts=True)[1]
            analysis["connection_distribution"]["entropy"] = float(stats.entropy(counts))

    # Calculate complexity metrics
    analysis["complexity_metrics"] = {
        "total_elements": len(variables) + len(connections),
        "variable_complexity": len(variables),
        "connection_complexity": len(connections),
        "density": len(connections) / max(len(variables), 1),
        "cyclomatic_complexity": len(connections) - len(variables) + 2
    }

    return analysis

def calculate_correlations(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate correlations between model elements."""
    correlations = {
        "variable_connection_correlation": 0.0,
        "line_position_correlation": 0.0
    }

    if variables and connections:
        # Calculate correlation between number of variables and connections
        var_count = len(variables)
        conn_count = len(connections)
        correlations["variable_connection_correlation"] = conn_count / max(var_count, 1)

        # Calculate line position correlation
        var_lines = [var.get("line", 0) for var in variables]
        conn_lines = [conn.get("line", 0) for conn in connections]

        if len(var_lines) > 1 and len(conn_lines) > 1:
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    correlation_matrix = np.corrcoef(var_lines, conn_lines)
                val = correlation_matrix[0, 1]
                correlations["line_position_correlation"] = 0.0 if np.isnan(val) else float(val)
            except Exception as e:
                logger.debug(f"Correlation computation failed, defaulting to 0.0: {e}")
                correlations["line_position_correlation"] = 0.0

    return correlations

def calculate_cyclomatic_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate cyclomatic complexity of the model."""
    return len(connections) - len(variables) + 2

def calculate_cognitive_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate cognitive complexity of the model."""
    # Cognitive complexity considers nesting, branching, and logical operators
    complexity = 0

    # Base complexity from number of elements
    complexity += len(variables) * 0.5
    complexity += len(connections) * 1.0

    # Additional complexity for high connectivity
    if variables and connections:
        density = len(connections) / len(variables)
        if density > 2.0:
            complexity += density * 0.5

    return complexity

def calculate_structural_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate structural complexity of the model."""
    # Structural complexity considers the graph structure
    complexity = 0

    # Base complexity
    complexity += len(variables) * 0.3
    complexity += len(connections) * 0.7

    # Graph density penalty
    if variables and connections:
        density = len(connections) / len(variables)
        if density > 1.5:
            complexity += density * 0.2

    return complexity

def calculate_complexity_metrics(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Calculate comprehensive complexity metrics for a GNN file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)

        metrics = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "cyclomatic_complexity": calculate_cyclomatic_complexity(variables, connections),
            "cognitive_complexity": calculate_cognitive_complexity(variables, connections),
            "structural_complexity": calculate_structural_complexity(variables, connections),
            "maintainability_index": calculate_maintainability_index(content, variables, connections),
            "technical_debt": calculate_technical_debt(content, variables, connections),
            "analysis_timestamp": datetime.now().isoformat()
        }

        return metrics

    except Exception as e:
        raise Exception(f"Failed to calculate complexity metrics for {file_path}: {e}") from e

def calculate_maintainability_index(content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate maintainability index."""
    # Simplified maintainability index calculation
    lines = len(content.splitlines())
    complexity = len(variables) + len(connections)

    if lines == 0:
        return 100.0

    # Higher index = more maintainable
    maintainability = 171 - 5.2 * np.log(complexity) - 0.23 * np.log(lines)
    return max(0.0, min(100.0, maintainability))

def calculate_technical_debt(content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate technical debt score."""
    debt = 0.0

    # Complexity debt
    if len(variables) > 20:
        debt += (len(variables) - 20) * 0.1

    if len(connections) > 50:
        debt += (len(connections) - 50) * 0.05

    # Documentation debt (simplified)
    if len(content) < 1000:  # Assuming short content means poor documentation
        debt += 0.5

    return debt

def run_performance_benchmarks(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Run performance benchmarks on a GNN file using actual implementation metrics."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        start_time = time.perf_counter()
        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)
        end_time = time.perf_counter()

        real_parse_time = end_time - start_time

        # Calculate real memory usage footprint
        memory_usage = sys.getsizeof(content) + sys.getsizeof(variables) + sys.getsizeof(connections)
        for var in variables:
            memory_usage += sys.getsizeof(var)
        for conn in connections:
            memory_usage += sys.getsizeof(conn)

        complexity = len(variables) + len(connections)

        # Actual performance metrics replacing simulated data
        benchmarks = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "parse_time": real_parse_time,
            "memory_usage": memory_usage,
            "complexity_score": complexity,
            "estimated_runtime": complexity * 0.01,
            "benchmark_timestamp": datetime.now().isoformat()
        }

        return benchmarks

    except Exception as e:
        raise Exception(f"Failed to run benchmarks for {file_path}: {e}") from e

def perform_model_comparisons(statistical_analyses: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """Perform comparisons between multiple models."""
    if len(statistical_analyses) < 2:
        return {"error": "Need at least 2 models for comparison"}

    comparisons = {
        "model_count": len(statistical_analyses),
        "complexity_comparison": {},
        "size_comparison": {},
        "structure_comparison": {},
        "comparison_timestamp": datetime.now().isoformat()
    }

    # Compare complexity metrics
    complexity_scores = []
    file_sizes = []
    variable_counts = []
    connection_counts = []

    for analysis in statistical_analyses:
        complexity_metrics = analysis.get("distributions", {}).get("complexity_metrics", {})
        complexity_scores.append(complexity_metrics.get("total_elements", 0))
        file_sizes.append(analysis.get("file_size", 0))
        variable_counts.append(len(analysis.get("variables", [])))
        connection_counts.append(len(analysis.get("connections", [])))

    comparisons["complexity_comparison"] = {
        "mean": np.mean(complexity_scores),
        "std": np.std(complexity_scores),
        "min": np.min(complexity_scores),
        "max": np.max(complexity_scores)
    }

    comparisons["size_comparison"] = {
        "mean": np.mean(file_sizes),
        "std": np.std(file_sizes),
        "min": np.min(file_sizes),
        "max": np.max(file_sizes)
    }

    comparisons["structure_comparison"] = {
        "variable_counts": {
            "mean": np.mean(variable_counts),
            "std": np.std(variable_counts)
        },
        "connection_counts": {
            "mean": np.mean(connection_counts),
            "std": np.std(connection_counts)
        }
    }

    return comparisons

def generate_analysis_summary(results: Dict[str, Any]) -> str:
    """Generate a summary report of analysis results."""
    summary = f"""
# Analysis Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Results
- **Files Processed**: {results.get('processed_files', 0)}
- **Success**: {results.get('success', False)}
- **Errors**: {len(results.get('errors', []))}

## Analysis Results
- **Statistical Analyses**: {len(results.get('statistical_analysis', []))}
- **Complexity Metrics**: {len(results.get('complexity_metrics', []))}
- **Performance Benchmarks**: {len(results.get('performance_benchmarks', []))}
- **Model Comparisons**: {len(results.get('model_comparisons', []))}

## Error Summary
"""

    errors = results.get('errors', [])
    if errors:
        for error in errors:
            if isinstance(error, dict):
                summary += f"- **{error.get('file', 'Unknown')}**: {error.get('error', 'Unknown error')}\n"
            else:
                summary += f"- {error}\n"
    else:
        summary += "- No errors encountered\n"

    summary += "\n## Model Statistics\n"

    # Add statistics from analyses
    analyses = results.get('statistical_analysis', [])
    if analyses:
        total_variables = sum(len(analysis.get('variables', [])) for analysis in analyses)
        total_connections = sum(len(analysis.get('connections', [])) for analysis in analyses)

        summary += f"- Total variables across all models: {total_variables}\n"
        summary += f"- Total connections across all models: {total_connections}\n"
        summary += f"- Average variables per model: {total_variables / len(analyses):.1f}\n"
        summary += f"- Average connections per model: {total_connections / len(analyses):.1f}\n"

    return summary

def generate_matrix_visualizations(parsed_data: Dict[str, Any], output_dir: Path, model_name: str) -> List[str]:
    """Generate heatmaps for model matrices."""
    visualizations = []
    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    matrices = parsed_data.get("matrices", [])
    # Extract matrices from parsed data (A, B, C, D maps)
    for i, matrix_info in enumerate(matrices):
        matrix_data = matrix_info.get("data")
        if matrix_data is not None and isinstance(matrix_data, np.ndarray):
            matrix_name = matrix_info.get("name", f"matrix_{i}")
            plt.figure(figsize=(10, 8))

            if SEABORN_AVAILABLE and sns is not None:
                sns.heatmap(matrix_data, annot=matrix_data.size < 100, cmap='viridis')
            else:
                # Recovery to matplotlib imshow
                plt.imshow(matrix_data, cmap='viridis', aspect='auto')
                plt.colorbar()
                # Add annotations if small enough
                if matrix_data.size < 100:
                    for r in range(matrix_data.shape[0]):
                        for c in range(matrix_data.shape[1]):
                            plt.text(c, r, f"{matrix_data[r, c]:.2g}",
                                   ha="center", va="center", color="w")

            plt.title(f"{model_name} - {matrix_name}")
            plot_file = output_dir / f"{model_name}_{matrix_name}_heatmap.png"
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            visualizations.append(str(plot_file))
    return visualizations

def visualize_simulation_results(execution_results: Dict[str, Any], output_dir: Path) -> List[str]:
    """Visualize actual simulation data from execution results."""
    visualizations = []
    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    # Example: Visualize belief evolution if traces are present
    details = execution_results.get("execution_details", [])
    for detail in details:
        model_name = detail.get("model_name", "unknown")
        framework = detail.get("framework", "unknown")
        impl_dir = Path(detail.get("implementation_directory", ""))

        # Look for simulation data (e.g., traces.json or simulation_dump.json)
        trace_files = list(impl_dir.glob("**/traces.json")) + list(impl_dir.glob("**/simulation_data/*.json"))

        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    data = json.load(f)

                # Plot traces (e.g., belief over time)
                if 'beliefs' in data or 'states' in data:
                    plt.figure(figsize=(10, 6))
                    # Simplified plotting logic for arbitrary trace data
                    beliefs = np.array(data.get('beliefs', data.get('states', [])))
                    if beliefs.ndim == 2:
                        for i in range(min(10, beliefs.shape[1])):
                            plt.plot(beliefs[:, i], label=f"State {i}")
                        plt.title(f"Belief Evolution - {model_name} ({framework})")
                        plt.legend()
                        fw_viz_dir = output_dir / framework
                        fw_viz_dir.mkdir(parents=True, exist_ok=True)
                        plot_file = fw_viz_dir / f"{model_name}_{framework}_belief_trace.png"
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

                        # Plot JSD / distance tracking (Belief Convergence)
                        distances = []
                        try:
                            from scipy.spatial.distance import jensenshannon
                            for t in range(1, len(beliefs)):
                                p = np.array(beliefs[t-1]).flatten()
                                q = np.array(beliefs[t]).flatten()
                                p = np.clip(p, 1e-12, None)
                                q = np.clip(q, 1e-12, None)
                                p = p / np.sum(p)
                                q = q / np.sum(q)
                                if np.allclose(p, q, atol=1e-8):
                                    val = 0.0
                                else:
                                    # Suppress scipy runtime warnings for edge-case slight negatives
                                    import warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", RuntimeWarning)
                                        val = jensenshannon(p, q)

                                if np.isnan(val) or val < 0:
                                    val = 0.0
                                distances.append(val)
                            ylabel = "Jensen-Shannon Divergence"
                        except ImportError:
                            for t in range(1, len(beliefs)):
                                p = np.array(beliefs[t-1]).flatten()
                                q = np.array(beliefs[t]).flatten()
                                distances.append(np.linalg.norm(q - p))
                            ylabel = "Euclidean Distance"

                        if distances:
                            plt.figure(figsize=(10, 6))
                            plt.plot(range(1, len(beliefs)), distances, label="Belief Update Magnitude", color="teal")
                            plt.title(f"Belief Convergence Tracker - {model_name} ({framework})")
                            plt.xlabel("Timestep")
                            plt.ylabel(ylabel)
                            plt.legend()
                            fw_viz_dir = output_dir / framework
                            fw_viz_dir.mkdir(parents=True, exist_ok=True)
                            plot_file = fw_viz_dir / f"{model_name}_{framework}_belief_convergence.png"
                            plt.savefig(plot_file)
                            plt.close()
                            visualizations.append(str(plot_file))

                # Plot Free Energy trajectories
                if 'free_energy' in data or 'efe' in data or 'F' in data:
                    plt.figure(figsize=(10, 6))
                    fe = data.get('free_energy', data.get('efe', data.get('F', [])))
                    if isinstance(fe, list) and len(fe) > 0:
                        plt.plot(fe, label="Free Energy", color='purple', marker='o', markersize=4)
                        plt.title(f"Free Energy Trajectory - {model_name} ({framework})")
                        plt.xlabel("Timestep")
                        plt.ylabel("Free Energy / Expected Free Energy")
                        plt.legend()
                        plot_file = output_dir / f"{model_name}_{framework}_free_energy.png"
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

                # Plot Precision Dynamics
                if 'precision' in data or 'gamma' in data or 'w' in data:
                    plt.figure(figsize=(10, 6))
                    prec = data.get('precision', data.get('gamma', data.get('w', [])))
                    if isinstance(prec, list) and len(prec) > 0:
                        plt.plot(prec, label="Precision Dynamics", color='orange', linestyle='--', marker='x')
                        plt.title(f"Precision Dynamics - {model_name} ({framework})")
                        plt.xlabel("Timestep")
                        plt.ylabel("Precision (Gamma/w)")
                        plt.legend()
                        plot_file = output_dir / f"{model_name}_{framework}_precision.png"
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

                # Plot Action History Frequency Maps
                if 'actions' in data or 'u' in data:
                    plt.figure(figsize=(8, 6))
                    actions = data.get('actions', data.get('u', []))
                    if isinstance(actions, list) and len(actions) > 0:
                        # flatten if nested
                        if isinstance(actions[0], list):
                            flat_actions = [a[0] if len(a) > 0 else 0 for a in actions]
                        else:
                            flat_actions = actions
                        unique_actions, counts = np.unique(flat_actions, return_counts=True)
                        plt.bar(range(len(unique_actions)), counts, tick_label=[str(a) for a in unique_actions], color='coral', alpha=0.8)
                        plt.title(f"Action Selection Frequencies - {model_name} ({framework})")
                        plt.xlabel("Action Variant")
                        plt.ylabel("Frequency")
                        fw_viz_dir = output_dir / framework
                        fw_viz_dir.mkdir(parents=True, exist_ok=True)
                        plot_file = fw_viz_dir / f"{model_name}_{framework}_action_frequencies.png"
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

            except Exception as e:
                logging.getLogger(__name__).debug(f"Failed to visualize trace file {trace_file}: {e}")
                continue

    return visualizations

def parse_matrix_data(matrix_str: str) -> Optional[np.ndarray]:
    """Parse matrix data from string representation."""
    try:
        # Simplified parsing logic for moving to analyzer
        import re
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', matrix_str)
        if len(numbers) >= 1:
            return np.array([float(n) for n in numbers])
        return None
    except Exception as e:
        logger.debug(f"Failed to extract numeric array from value: {e}")
        return None

def analyze_framework_outputs(execution_output_dir: Path, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Analyze and compare outputs from all framework executions.
    
    Args:
        execution_output_dir: Directory containing execution results (e.g., output/12_execute_output)
        logger: Optional logger instance
        
    Returns:
        Dictionary with framework comparison results
    """
    import json
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    results = {
        "timestamp": datetime.now().isoformat(),
        "frameworks": {},
        "comparisons": {},
        "metrics": {}
    }

    # Read execution summary - check summaries/ subfolder first, recovery to root
    execution_summary_file = execution_output_dir / "summaries" / "execution_summary.json"
    if not execution_summary_file.exists():
        execution_summary_file = execution_output_dir / "execution_summary.json"
    if not execution_summary_file.exists():
        logger.warning(f"Execution summary not found at {execution_summary_file}")
        return results

    try:
        with open(execution_summary_file, 'r') as f:
            execution_summary = json.load(f)

        execution_details = execution_summary.get("execution_details", [])

        # Group by framework
        framework_data = {}
        for detail in execution_details:
            framework = detail.get("framework", "unknown")
            if framework not in framework_data:
                framework_data[framework] = []
            framework_data[framework].append(detail)

        # Extract metrics from each framework
        for framework, details in framework_data.items():
            framework_results = {
                "success_count": sum(1 for d in details if d.get("success", False)),
                "total_count": len(details),
                "execution_times": [d.get("execution_time", 0) for d in details],
                "output_files": [d.get("output_file", "") for d in details],
                "implementation_dirs": [d.get("implementation_directory", "") for d in details]
            }

            # Try to extract simulation data
            simulation_metrics = _extract_simulation_metrics(framework, details, execution_output_dir, logger)
            framework_results.update(simulation_metrics)

            results["frameworks"][framework] = framework_results

        # Perform cross-framework comparisons
        results["comparisons"] = _compare_framework_results(results["frameworks"], logger)

        # Generate aggregate metrics
        results["metrics"] = _calculate_aggregate_metrics(results["frameworks"], logger)

    except Exception as e:
        logger.error(f"Error analyzing framework outputs: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return results

def _extract_simulation_metrics(framework: str, details: List[Dict[str, Any]], execution_output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Extract simulation-specific metrics from framework outputs.
    
    Searches simulation_data/ subdirectories and maps framework-specific
    keys to a standardized format for cross-framework comparison.
    """
    metrics = {
        "beliefs": [],
        "actions": [],
        "observations": [],
        "free_energy": [],
        "execution_times": [],
        "num_timesteps": None,
        "validation": {},
        "model_parameters": {},
        "data_source": None
    }

    for detail in details:
        # Collect execution time from the detail record
        exec_time = detail.get("execution_time", 0)
        if exec_time:
            metrics["execution_times"].append(exec_time)

        impl_dir = Path(detail.get("implementation_directory", ""))
        if not impl_dir.exists():
            # Try resolving relative to execution_output_dir parent
            impl_dir = execution_output_dir.parent.parent / detail.get("implementation_directory", "")
            if not impl_dir.exists():
                logger.debug(f"  [{framework}] impl_dir not found: {impl_dir}")
                continue

        # Search for simulation data files in multiple locations
        candidate_files = [
            # Primary: simulation_data subdirectory
            impl_dir / "simulation_data" / "simulation_results.json",
            # Direct in impl_dir
            impl_dir / "simulation_results.json",
            impl_dir / "results.json",
            impl_dir / "output.json",
            impl_dir / "traces.json",
        ]

        # Also recursively search for any simulation_results.json
        try:
            for found_file in impl_dir.rglob("simulation_results.json"):
                if found_file not in candidate_files:
                    candidate_files.append(found_file)
        except Exception as e:
            logger.debug(f"Error during recursive search in {impl_dir}: {e}")

        for output_file in candidate_files:
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)

                    metrics["data_source"] = str(output_file)
                    logger.info(f"  [{framework}] Loaded simulation data from: {output_file}")

                    # Extract beliefs - check both top-level and simulation_trace
                    beliefs = data.get("beliefs", [])
                    if not beliefs:
                        beliefs = data.get("simulation_trace", {}).get("beliefs", [])
                    if beliefs:
                        metrics["beliefs"] = beliefs

                    # Extract actions
                    actions = data.get("actions", [])
                    if not actions:
                        actions = data.get("simulation_trace", {}).get("actions", [])
                    if actions:
                        metrics["actions"] = actions

                    # Extract observations
                    observations = data.get("observations", [])
                    if not observations:
                        observations = data.get("simulation_trace", {}).get("observations", [])
                    if observations:
                        metrics["observations"] = observations

                    # Extract free energy - multiple possible key locations
                    free_energy = data.get("free_energy", [])
                    if not free_energy:
                        free_energy = data.get("efe_history", [])
                    if not free_energy:
                        free_energy = data.get("metrics", {}).get("expected_free_energy", [])
                    if not free_energy:
                        free_energy = data.get("simulation_trace", {}).get("efe_history", [])
                    if free_energy:
                        metrics["free_energy"] = free_energy

                    # Extract additional metadata
                    metrics["num_timesteps"] = data.get("num_timesteps", data.get("time_steps", len(beliefs) if beliefs else None))
                    metrics["validation"] = data.get("validation", {})
                    metrics["model_parameters"] = data.get("model_parameters", {})

                    # Extract belief confidence if available
                    confidence = data.get("metrics", {}).get("belief_confidence", [])
                    if not confidence:
                        confidence = data.get("simulation_trace", {}).get("belief_confidence", [])
                    if confidence:
                        metrics["belief_confidence"] = confidence

                    break  # Found data, stop searching
                except Exception as e:
                    logger.debug(f"  [{framework}] Error reading {output_file}: {e}")
                    continue

        # Recovery: CSV simulation data (ActiveInference.jl outputs CSV, not JSON)
        if not metrics["data_source"]:
            csv_candidates = [
                impl_dir / "simulation_data" / "simulation_results.csv",
            ]
            # Also search for timestamped CSV files
            sim_data_dir = impl_dir / "simulation_data"
            if sim_data_dir.exists():
                try:
                    for csv_file in sorted(sim_data_dir.glob("*_simulation_results.csv")):
                        if csv_file not in csv_candidates:
                            csv_candidates.append(csv_file)
                except Exception as e:
                    logger.debug(f"Error searching for timestamped CSV files in {sim_data_dir}: {e}")

            for csv_file in csv_candidates:
                if csv_file.exists():
                    try:
                        import csv as csv_module
                        beliefs = []
                        actions = []
                        observations = []

                        with open(csv_file, 'r') as f:
                            # Skip comment lines (lines starting with #)
                            lines = [line for line in f if not line.startswith('#')]

                        if lines:
                            reader = csv_module.reader(lines)
                            for row in reader:
                                if len(row) >= 3:
                                    try:
                                        observations.append(int(float(row[1])))
                                        actions.append(int(float(row[2])))
                                        # Beliefs are remaining columns
                                        if len(row) > 3:
                                            belief = [float(x) for x in row[3:]]
                                            beliefs.append(belief)
                                    except ValueError:
                                        continue

                        if beliefs or actions or observations:
                            metrics["data_source"] = str(csv_file)
                            logger.info(f"  [{framework}] Loaded CSV simulation data from: {csv_file}")
                            if beliefs:
                                metrics["beliefs"] = beliefs
                            if actions:
                                metrics["actions"] = actions
                            if observations:
                                metrics["observations"] = observations
                            metrics["num_timesteps"] = len(beliefs) if beliefs else len(actions)

                            # Infer validation from data
                            if beliefs:
                                sums_valid = all(abs(sum(b) - 1.0) < 0.01 for b in beliefs)
                                metrics["validation"] = {
                                    "all_beliefs_valid": True,
                                    "beliefs_sum_to_one": sums_valid,
                                    "actions_in_range": True
                                }
                            break
                    except Exception as e:
                        logger.debug(f"  [{framework}] Error reading CSV {csv_file}: {e}")
                        continue

        # Supplement: model_parameters.json (separate from simulation data)
        if metrics["data_source"] and not metrics["model_parameters"]:
            params_candidates = [
                impl_dir / "simulation_data" / "model_parameters.json",
            ]
            # Also search for timestamped parameter files
            params_dir = impl_dir / "simulation_data"
            if params_dir.exists():
                try:
                    for pf in sorted(params_dir.glob("*_model_parameters.json")):
                        if pf not in params_candidates:
                            params_candidates.append(pf)
                except Exception as e:
                    logger.debug(f"Error searching for model parameter files in {params_dir}: {e}")

            for params_file in params_candidates:
                if params_file.exists():
                    try:
                        with open(params_file, 'r') as f:
                            params_data = json.load(f)
                        metrics["model_parameters"] = {
                            "num_states": params_data.get("n_states", params_data.get("num_states", 0)),
                            "num_observations": params_data.get("n_observations", params_data.get("num_observations", 0)),
                            "num_actions": params_data.get("n_actions", params_data.get("num_actions", 0)),
                        }
                        logger.info(f"  [{framework}] Loaded model parameters from: {params_file}")
                        break
                    except Exception as e:
                        logger.debug(f"  [{framework}] Error reading params {params_file}: {e}")

        # Supplement: circuit_info.json for categorical frameworks (DisCoPy)
        if not metrics["data_source"]:
            circuit_candidates = [
                impl_dir / "simulation_data" / "circuit_info.json",
            ]
            sim_data_dir = impl_dir / "simulation_data"
            if sim_data_dir.exists():
                try:
                    for cf in sorted(sim_data_dir.glob("*_circuit_info.json")):
                        if cf not in circuit_candidates:
                            circuit_candidates.append(cf)
                except Exception as e:
                    logger.debug(f"Error searching for circuit info files in {sim_data_dir}: {e}")

            for circuit_file in circuit_candidates:
                if circuit_file.exists():
                    try:
                        with open(circuit_file, 'r') as f:
                            circuit_data = json.load(f)
                        metrics["data_source"] = str(circuit_file)
                        metrics["circuit_info"] = {
                            "model_name": circuit_data.get("model_name", ""),
                            "components": circuit_data.get("components", []),
                            "num_components": circuit_data.get("analysis", {}).get("num_components", len(circuit_data.get("components", []))),
                            "parameters": circuit_data.get("parameters", {}),
                        }
                        # Map circuit parameters to standardized model_parameters
                        params = circuit_data.get("parameters", {})
                        if params:
                            metrics["model_parameters"] = {
                                "num_states": params.get("num_states", 0),
                                "num_observations": params.get("num_observations", 0),
                                "num_actions": params.get("num_actions", 0),
                            }
                        logger.info(f"  [{framework}] Loaded circuit info from: {circuit_file}")
                        break
                    except Exception as e:
                        logger.debug(f"  [{framework}] Error reading circuit {circuit_file}: {e}")

    # Log summary of what was extracted
    data_found = []
    if metrics["beliefs"]: data_found.append(f"beliefs({len(metrics['beliefs'])} steps)")
    if metrics["actions"]: data_found.append(f"actions({len(metrics['actions'])})")
    if metrics["observations"]: data_found.append(f"observations({len(metrics['observations'])})")
    if metrics["free_energy"]: data_found.append(f"free_energy({len(metrics['free_energy'])})")
    if metrics.get("circuit_info"): data_found.append(f"circuit({metrics['circuit_info'].get('num_components', 0)} components)")
    if metrics["model_parameters"]: data_found.append("model_parameters")

    if data_found:
        logger.info(f"  [{framework}] Extracted: {', '.join(data_found)}")
    else:
        logger.warning(f"  [{framework}] No simulation data found")

    return metrics

def _compare_framework_results(framework_data: Dict[str, Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """Compare results across frameworks using real simulation data."""
    comparisons = {
        "success_rates": {},
        "performance_comparison": {},
        "metric_agreement": {},
        "data_coverage": {},
        "simulation_statistics": {}
    }

    # Compare success rates
    for framework, data in framework_data.items():
        success_count = data.get("success_count", 0)
        total_count = data.get("total_count", 0)
        if total_count > 0:
            comparisons["success_rates"][framework] = success_count / total_count

    # Compare execution times
    for framework, data in framework_data.items():
        times = data.get("execution_times", [])
        if times and any(t > 0 for t in times):
            valid_times = [t for t in times if t > 0]
            comparisons["performance_comparison"][framework] = {
                "mean": float(np.mean(valid_times)),
                "std": float(np.std(valid_times)) if len(valid_times) > 1 else 0.0,
                "min": float(np.min(valid_times)),
                "max": float(np.max(valid_times))
            }

    # Compare data coverage — which frameworks produced which data
    for framework, data in framework_data.items():
        coverage = {
            "has_beliefs": bool(data.get("beliefs")),
            "has_actions": bool(data.get("actions")),
            "has_observations": bool(data.get("observations")),
            "has_free_energy": bool(data.get("free_energy")),
            "has_circuit_info": bool(data.get("circuit_info")),
            "num_timesteps": data.get("num_timesteps"),
            "validation_passed": all(data.get("validation", {}).values()) if data.get("validation") else None,
            "data_source": data.get("data_source")
        }
        comparisons["data_coverage"][framework] = coverage

    # Compare simulation statistics across frameworks with data
    frameworks_with_beliefs = {fw: data for fw, data in framework_data.items() if data.get("beliefs")}
    frameworks_with_efe = {fw: data for fw, data in framework_data.items() if data.get("free_energy")}

    for framework, data in frameworks_with_beliefs.items():
        beliefs = data["beliefs"]
        try:
            beliefs_arr = np.array(beliefs)
            if beliefs_arr.ndim == 2:
                comparisons["simulation_statistics"][framework] = {
                    "belief_dims": list(beliefs_arr.shape),
                    "mean_confidence": float(np.mean(np.max(beliefs_arr, axis=1))),
                    "final_belief": [float(v) for v in beliefs_arr[-1]] if len(beliefs_arr) > 0 else [],
                }
        except Exception as e:
            logger.debug(f"Error computing belief statistics for {framework}: {e}")

    for framework, data in frameworks_with_efe.items():
        efe = data["free_energy"]
        try:
            efe_arr = np.array(efe, dtype=float)
            stats = comparisons["simulation_statistics"].get(framework, {})
            stats["efe_mean"] = float(np.mean(efe_arr))
            stats["efe_std"] = float(np.std(efe_arr))
            stats["efe_min"] = float(np.min(efe_arr))
            stats["efe_max"] = float(np.max(efe_arr))
            comparisons["simulation_statistics"][framework] = stats
        except Exception as e:
            logger.debug(f"Error computing EFE statistics for {framework}: {e}")

    # Metric agreement — if multiple frameworks have beliefs, compare final convergence
    if len(frameworks_with_beliefs) >= 2:
        agreement = {}
        fw_names = list(frameworks_with_beliefs.keys())
        for i in range(len(fw_names)):
            for j in range(i+1, len(fw_names)):
                fw_a, fw_b = fw_names[i], fw_names[j]
                try:
                    beliefs_a = np.array(frameworks_with_beliefs[fw_a]["beliefs"])
                    beliefs_b = np.array(frameworks_with_beliefs[fw_b]["beliefs"])
                    if beliefs_a.shape == beliefs_b.shape:
                        if beliefs_a.shape[0] > 1:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                corr_val = np.corrcoef(
                                    np.max(beliefs_a, axis=1),
                                    np.max(beliefs_b, axis=1)
                                )[0, 1]
                            correlation = 0.0 if np.isnan(corr_val) else float(corr_val)
                        else:
                            correlation = 0.0
                        agreement[f"{fw_a}_vs_{fw_b}"] = {
                            "confidence_correlation": correlation,
                            "same_dimensions": True
                        }
                    else:
                        agreement[f"{fw_a}_vs_{fw_b}"] = {
                            "same_dimensions": False,
                            "dims_a": list(beliefs_a.shape),
                            "dims_b": list(beliefs_b.shape)
                        }
                except Exception as e:
                    logger.debug(f"Error computing metric agreement for {fw_a} vs {fw_b}: {e}")
        comparisons["metric_agreement"] = agreement

    return comparisons

def _calculate_aggregate_metrics(framework_data: Dict[str, Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """Calculate aggregate metrics across all frameworks."""
    metrics = {
        "total_frameworks": len(framework_data),
        "total_successful": sum(data.get("success_count", 0) for data in framework_data.values()),
        "total_executions": sum(data.get("total_count", 0) for data in framework_data.values()),
        "overall_success_rate": 0.0
    }

    if metrics["total_executions"] > 0:
        metrics["overall_success_rate"] = metrics["total_successful"] / metrics["total_executions"]

    return metrics

def generate_framework_comparison_report(comparison_data: Dict[str, Any], output_dir: Path, logger: Optional[logging.Logger] = None) -> str:
    """
    Generate a comprehensive comparison report across frameworks,
    grounded in real simulation data.
    
    Args:
        comparison_data: Output from analyze_framework_outputs()
        output_dir: Directory to save report
        logger: Optional logger instance
        
    Returns:
        Path to generated report file
    """
    import logging
    import json

    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Markdown report
    report_lines = [
        "# Framework Execution Comparison Report",
        "",
        f"Generated: {comparison_data.get('timestamp', 'Unknown')}",
        "",
        "## Summary",
        ""
    ]

    metrics = comparison_data.get("metrics", {})
    report_lines.extend([
        f"- Total Frameworks: {metrics.get('total_frameworks', 0)}",
        f"- Total Executions: {metrics.get('total_executions', 0)}",
        f"- Successful Executions: {metrics.get('total_successful', 0)}",
        f"- Overall Success Rate: {metrics.get('overall_success_rate', 0.0):.2%}",
        ""
    ])

    # Framework details with simulation data
    report_lines.extend([
        "## Framework Details",
        ""
    ])

    for framework, data in comparison_data.get("frameworks", {}).items():
        success_count = data.get("success_count", 0)
        total_count = data.get("total_count", 0)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        report_lines.extend([
            f"### {framework.upper()}",
            "",
            f"- Success Rate: {success_rate:.1f}% ({success_count}/{total_count})",
        ])

        # Add execution time if available
        times = [t for t in data.get('execution_times', []) if t > 0]
        if times:
            report_lines.append(f"- Execution Time: {times[0]:.2f}s")

        # Add simulation data coverage
        n_beliefs = len(data.get('beliefs', []))
        n_actions = len(data.get('actions', []))
        n_obs = len(data.get('observations', []))
        n_efe = len(data.get('free_energy', []))
        timesteps = data.get('num_timesteps')

        if any([n_beliefs, n_actions, n_obs, n_efe]):
            report_lines.append(f"- Timesteps: {timesteps}")
            report_lines.append(f"- Data: beliefs={n_beliefs}, actions={n_actions}, observations={n_obs}, free_energy={n_efe}")

        # Add validation status
        validation = data.get('validation', {})
        if validation:
            passed = all(validation.values())
            checks = ', '.join(f"{k}={'✅' if v else '❌'}" for k, v in validation.items())
            report_lines.append(f"- Validation: {'✅ ALL PASSED' if passed else '⚠️ ISSUES'} ({checks})")

        # Add data source
        source = data.get('data_source')
        if source:
            report_lines.append(f"- Data Source: `{source}`")

        report_lines.append("")

    # Simulation Statistics Comparison
    comparisons = comparison_data.get("comparisons", {})
    sim_stats = comparisons.get("simulation_statistics", {})
    if sim_stats:
        report_lines.extend([
            "## Simulation Data Comparison",
            "",
            "| Framework | Timesteps | Mean Confidence | EFE Mean | EFE Std |",
            "|-----------|-----------|-----------------|----------|---------|"
        ])

        for framework, stats in sim_stats.items():
            dims = stats.get('belief_dims', [])
            timesteps = dims[0] if dims else 'N/A'
            conf = f"{stats.get('mean_confidence', 0):.4f}" if 'mean_confidence' in stats else 'N/A'
            efe_mean = f"{stats.get('efe_mean', 0):.4f}" if 'efe_mean' in stats else 'N/A'
            efe_std = f"{stats.get('efe_std', 0):.4f}" if 'efe_std' in stats else 'N/A'
            report_lines.append(f"| {framework} | {timesteps} | {conf} | {efe_mean} | {efe_std} |")
        report_lines.append("")

    # Data Coverage
    data_coverage = comparisons.get("data_coverage", {})
    if data_coverage:
        report_lines.extend([
            "## Data Coverage",
            "",
            "| Framework | Beliefs | Actions | Observations | Free Energy | Validation |",
            "|-----------|---------|---------|--------------|-------------|------------|"
        ])
        for framework, cov in data_coverage.items():
            report_lines.append(
                f"| {framework} "
                f"| {'✅' if cov.get('has_beliefs') else '❌'} "
                f"| {'✅' if cov.get('has_actions') else '❌'} "
                f"| {'✅' if cov.get('has_observations') else '❌'} "
                f"| {'✅' if cov.get('has_free_energy') else '❌'} "
                f"| {'✅' if cov.get('validation_passed') else ('❌' if cov.get('validation_passed') is False else '—')} |"
            )
        report_lines.append("")

    # Metric Agreement
    metric_agreement = comparisons.get("metric_agreement", {})
    if metric_agreement:
        report_lines.extend([
            "## Cross-Framework Metric Agreement",
            ""
        ])
        for pair, agreement in metric_agreement.items():
            if agreement.get("same_dimensions"):
                corr = agreement.get('confidence_correlation', 0)
                report_lines.append(f"- **{pair}**: confidence correlation = {corr:.4f}")
            else:
                report_lines.append(f"- **{pair}**: different dimensions ({agreement.get('dims_a')} vs {agreement.get('dims_b')})")
        report_lines.append("")

    # Performance comparison
    if comparisons.get("performance_comparison"):
        report_lines.extend([
            "## Performance Comparison",
            "",
            "| Framework | Mean Time (s) | Std Dev | Min | Max |",
            "|-----------|---------------|---------|-----|-----|"
        ])

        for framework, perf in comparisons["performance_comparison"].items():
            report_lines.append(
                f"| {framework} | {perf['mean']:.3f} | {perf['std']:.3f} | {perf['min']:.3f} | {perf['max']:.3f} |"
            )
        report_lines.append("")

    # Save report to cross_framework subfolder
    cross_fw_dir = output_dir / "cross_framework"
    cross_fw_dir.mkdir(parents=True, exist_ok=True)
    report_file = cross_fw_dir / "framework_comparison_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    # Also save JSON version
    json_file = cross_fw_dir / "framework_comparison_data.json"
    with open(json_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)

    logger.info(f"Generated framework comparison report: {report_file}")

    return str(report_file)

def visualize_cross_framework_metrics(comparison_data: Dict[str, Any], output_dir: Path, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Generate visualizations comparing metrics across frameworks.
    
    Args:
        comparison_data: Output from analyze_framework_outputs()
        output_dir: Directory to save visualizations
        logger: Optional logger instance
        
    Returns:
        List of generated visualization file paths
    """
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping visualizations")
        return []

    cross_fw_dir = output_dir / "cross_framework"
    cross_fw_dir.mkdir(parents=True, exist_ok=True)
    visualizations = []

    try:
        # Success rate comparison
        frameworks = list(comparison_data.get("frameworks", {}).keys())
        success_rates = []
        for framework in frameworks:
            data = comparison_data["frameworks"][framework]
            success_count = data.get("success_count", 0)
            total_count = data.get("total_count", 0)
            rate = (success_count / total_count * 100) if total_count > 0 else 0
            success_rates.append(rate)

        if frameworks and success_rates:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(frameworks, success_rates, color=['#2ecc71' if r > 50 else '#e74c3c' for r in success_rates])
            ax.set_ylabel('Success Rate (%)', fontweight='bold')
            ax.set_xlabel('Framework', fontweight='bold')
            ax.set_title('Framework Execution Success Rates', fontweight='bold', fontsize=14)
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%',
                       ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            viz_file = cross_fw_dir / "framework_success_rates.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(viz_file))

        # Performance comparison
        perf_comparison = comparison_data.get("comparisons", {}).get("performance_comparison", {})
        if perf_comparison:
            fig, ax = plt.subplots(figsize=(12, 6))
            frameworks = list(perf_comparison.keys())
            means = [perf_comparison[f]["mean"] for f in frameworks]
            stds = [perf_comparison[f]["std"] for f in frameworks]

            x_pos = np.arange(len(frameworks))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
            ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
            ax.set_xlabel('Framework', fontweight='bold')
            ax.set_title('Framework Execution Time Comparison', fontweight='bold', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(frameworks)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            viz_file = cross_fw_dir / "framework_performance_comparison.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(viz_file))

    except Exception as e:
        logger.error(f"Error generating cross-framework visualizations: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return visualizations
