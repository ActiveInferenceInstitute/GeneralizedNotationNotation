#!/usr/bin/env python3
"""
Analysis analyzer module for GNN statistical analysis.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import numpy as np
from datetime import datetime

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
