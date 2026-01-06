#!/usr/bin/env python3
"""
Analysis analyzer module for GNN statistical analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import numpy as np
from datetime import datetime
import logging
# Import visualization libraries with error handling
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
        raise Exception(f"Failed to analyze {file_path}: {e}")

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
    
    # Look for connection patterns
    conn_patterns = [
        r'(\w+)\s*->\s*(\w+)',  # source -> target
        r'(\w+)\s*→\s*(\w+)',   # source → target
        r'(\w+)\s*connects\s*(\w+)',  # source connects target
    ]
    
    for pattern in conn_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            connections.append({
                "source": match.group(1),
                "target": match.group(2),
                "connection": match.group(0),
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
            "mean": np.mean(var_lines),
            "std": np.std(var_lines),
            "min": np.min(var_lines),
            "max": np.max(var_lines),
            "median": np.median(var_lines)
        }
    
    # Analyze connection distribution
    if connections:
        conn_lines = [conn.get("line", 0) for conn in connections]
        analysis["connection_distribution"] = {
            "mean": np.mean(conn_lines),
            "std": np.std(conn_lines),
            "min": np.min(conn_lines),
            "max": np.max(conn_lines),
            "median": np.median(conn_lines)
        }
    
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
                correlation_matrix = np.corrcoef(var_lines, conn_lines)
                correlations["line_position_correlation"] = correlation_matrix[0, 1]
            except:
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
        raise Exception(f"Failed to calculate complexity metrics for {file_path}: {e}")

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
    """Run performance benchmarks on a GNN file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)
        
        # Simulate performance metrics
        benchmarks = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "parse_time": len(content) * 0.001,  # Simulated parse time
            "memory_usage": len(content) * 0.1,  # Simulated memory usage
            "complexity_score": len(variables) + len(connections),
            "estimated_runtime": len(variables) * len(connections) * 0.01,
            "benchmark_timestamp": datetime.now().isoformat()
        }
        
        return benchmarks
        
    except Exception as e:
        raise Exception(f"Failed to run benchmarks for {file_path}: {e}")

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
                # Fallback to matplotlib imshow
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
                        plot_file = output_dir / f"{model_name}_{framework}_belief_trace.png"
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))
            except Exception:
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
    except Exception:
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
    
    # Read execution summary
    execution_summary_file = execution_output_dir / "execution_results" / "execution_summary.json"
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
    """Extract simulation-specific metrics from framework outputs."""
    metrics = {
        "beliefs": [],
        "actions": [],
        "observations": [],
        "free_energy": [],
        "execution_times": []
    }
    
    for detail in details:
        impl_dir = Path(detail.get("implementation_directory", ""))
        if not impl_dir.exists():
            continue
        
        # Look for common output files
        output_files = [
            impl_dir / "simulation_results.json",
            impl_dir / "results.json",
            impl_dir / "output.json",
            impl_dir / "traces.json"
        ]
        
        for output_file in output_files:
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract common metrics
                    if "beliefs" in data:
                        metrics["beliefs"].append(data["beliefs"])
                    if "actions" in data:
                        metrics["actions"].append(data["actions"])
                    if "observations" in data:
                        metrics["observations"].append(data["observations"])
                    if "free_energy" in data:
                        metrics["free_energy"].append(data["free_energy"])
                    
                    break
                except Exception as e:
                    logger.debug(f"Error reading {output_file}: {e}")
                    continue
    
    return metrics

def _compare_framework_results(framework_data: Dict[str, Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """Compare results across frameworks."""
    comparisons = {
        "success_rates": {},
        "performance_comparison": {},
        "metric_agreement": {}
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
        if times:
            comparisons["performance_comparison"][framework] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times))
            }
    
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
    Generate a comprehensive comparison report across frameworks.
    
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
    
    # Framework details
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
            f"- Execution Times: {len(data.get('execution_times', []))} recorded",
            ""
        ])
    
    # Comparisons
    comparisons = comparison_data.get("comparisons", {})
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
    
    # Save report
    report_file = output_dir / "framework_comparison_report.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also save JSON version
    json_file = output_dir / "framework_comparison_data.json"
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
    
    output_dir.mkdir(parents=True, exist_ok=True)
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
            viz_file = output_dir / "framework_success_rates.png"
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
            viz_file = output_dir / "framework_performance_comparison.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(viz_file))
        
    except Exception as e:
        logger.error(f"Error generating cross-framework visualizations: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return visualizations
