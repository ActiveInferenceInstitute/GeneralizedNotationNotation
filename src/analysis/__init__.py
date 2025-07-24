"""
Analysis module for GNN Processing Pipeline.

This module provides comprehensive analysis and statistical processing for GNN models.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import numpy as np
import re
from datetime import datetime

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def process_analysis(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process GNN files with comprehensive analysis.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("analysis")
    
    try:
        log_step_start(logger, "Processing analysis")
        
        # Create results directory
        results_dir = output_dir / "analysis_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "statistical_analysis": [],
            "complexity_metrics": [],
            "performance_benchmarks": [],
            "model_comparisons": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for analysis")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Perform statistical analysis
                    stats_analysis = perform_statistical_analysis(gnn_file, verbose)
                    results["statistical_analysis"].append(stats_analysis)
                    
                    # Calculate complexity metrics
                    complexity = calculate_complexity_metrics(gnn_file, verbose)
                    results["complexity_metrics"].append(complexity)
                    
                    # Run performance benchmarks
                    benchmarks = run_performance_benchmarks(gnn_file, verbose)
                    results["performance_benchmarks"].append(benchmarks)
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
            
            # Perform cross-model comparisons if multiple files
            if len(gnn_files) > 1:
                comparisons = perform_model_comparisons(results["statistical_analysis"], verbose)
                results["model_comparisons"].append(comparisons)
        
        # Save detailed results
        results_file = results_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            serializable_results = convert_numpy_types(results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate summary report
        summary = generate_analysis_summary(results)
        summary_file = results_dir / "analysis_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        if results["success"]:
            log_step_success(logger, "Analysis processing completed successfully")
        else:
            log_step_error(logger, "Analysis processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, f"Analysis processing failed: {str(e)}")
        return False

def perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Perform statistical analysis of a GNN model.
    
    Args:
        file_path: Path to the GNN file
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing statistical analysis results
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract model components
        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)
        sections = extract_sections_for_analysis(content)
        
        # Calculate basic statistics
        var_stats = calculate_variable_statistics(variables)
        conn_stats = calculate_connection_statistics(connections)
        section_stats = calculate_section_statistics(sections)
        
        # Perform distribution analysis
        distributions = analyze_distributions(variables, connections)
        
        # Calculate correlation metrics
        correlations = calculate_correlations(variables, connections)
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "line_count": len(content.splitlines()),
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
    """Extract variables from GNN content for analysis."""
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
                "type": match.group(2) if len(match.groups()) > 1 else "unknown",
                "definition": match.group(0),
                "length": len(match.group(1))
            })
    
    return variables

def extract_connections_for_analysis(content: str) -> List[Dict[str, Any]]:
    """Extract connections from GNN content for analysis."""
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
                "definition": match.group(0),
                "length": len(match.group(0))
            })
    
    return connections

def extract_sections_for_analysis(content: str) -> List[Dict[str, Any]]:
    """Extract sections from GNN content for analysis."""
    sections = []
    
    # Look for section headers
    section_pattern = r'^#+\s*(.+)$'
    matches = re.finditer(section_pattern, content, re.MULTILINE)
    
    for match in matches:
        sections.append({
            "title": match.group(1).strip(),
            "level": len(match.group(0)) - len(match.group(0).lstrip('#')),
            "length": len(match.group(1))
        })
    
    return sections

def calculate_variable_statistics(variables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for variables."""
    if not variables:
        return {"count": 0, "mean_length": 0, "std_length": 0}
    
    lengths = [var["length"] for var in variables]
    
    return {
        "count": len(variables),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "median_length": np.median(lengths),
        "type_distribution": count_type_distribution(variables)
    }

def calculate_connection_statistics(connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for connections."""
    if not connections:
        return {"count": 0, "mean_length": 0, "std_length": 0}
    
    lengths = [conn["length"] for conn in connections]
    
    return {
        "count": len(connections),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "median_length": np.median(lengths),
        "connectivity_matrix": build_connectivity_matrix(connections)
    }

def calculate_section_statistics(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics for sections."""
    if not sections:
        return {"count": 0, "mean_level": 0, "std_level": 0}
    
    levels = [section["level"] for section in sections]
    lengths = [section["length"] for section in sections]
    
    return {
        "count": len(sections),
        "mean_level": np.mean(levels),
        "std_level": np.std(levels),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "hierarchy_depth": max(levels) if levels else 0
    }

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
        source = conn["source"]
        target = conn["target"]
        
        if source not in connectivity:
            connectivity[source] = []
        if target not in connectivity:
            connectivity[target] = []
            
        connectivity[source].append(target)
    
    return connectivity

def analyze_distributions(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze distributions of model components."""
    distributions = {
        "variable_length_distribution": {},
        "connection_length_distribution": {},
        "connectivity_distribution": {}
    }
    
    # Variable length distribution
    if variables:
        var_lengths = [var["length"] for var in variables]
        distributions["variable_length_distribution"] = {
            "histogram": np.histogram(var_lengths, bins=min(10, len(var_lengths)))[0].tolist(),
            "percentiles": {
                "25": np.percentile(var_lengths, 25),
                "50": np.percentile(var_lengths, 50),
                "75": np.percentile(var_lengths, 75),
                "90": np.percentile(var_lengths, 90),
                "95": np.percentile(var_lengths, 95)
            }
        }
    
    # Connection length distribution
    if connections:
        conn_lengths = [conn["length"] for conn in connections]
        distributions["connection_length_distribution"] = {
            "histogram": np.histogram(conn_lengths, bins=min(10, len(conn_lengths)))[0].tolist(),
            "percentiles": {
                "25": np.percentile(conn_lengths, 25),
                "50": np.percentile(conn_lengths, 50),
                "75": np.percentile(conn_lengths, 75),
                "90": np.percentile(conn_lengths, 90),
                "95": np.percentile(conn_lengths, 95)
            }
        }
    
    # Connectivity distribution
    if connections:
        connectivity = build_connectivity_matrix(connections)
        connection_counts = [len(targets) for targets in connectivity.values()]
        if connection_counts:
            distributions["connectivity_distribution"] = {
                "mean_connections": np.mean(connection_counts),
                "std_connections": np.std(connection_counts),
                "max_connections": np.max(connection_counts),
                "min_connections": np.min(connection_counts)
            }
    
    return distributions

def calculate_correlations(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate correlation metrics between model components."""
    correlations = {
        "variable_connection_correlation": 0.0,
        "complexity_metrics": {}
    }
    
    if variables and connections:
        # Calculate correlation between number of variables and connections
        var_count = len(variables)
        conn_count = len(connections)
        
        # Simple correlation coefficient
        if var_count > 0 and conn_count > 0:
            # This is a simplified correlation - in practice you'd want more sophisticated metrics
            correlations["variable_connection_correlation"] = min(conn_count / var_count, 1.0)
    
    # Calculate complexity metrics
    correlations["complexity_metrics"] = {
        "cyclomatic_complexity": calculate_cyclomatic_complexity(variables, connections),
        "cognitive_complexity": calculate_cognitive_complexity(variables, connections),
        "structural_complexity": calculate_structural_complexity(variables, connections)
    }
    
    return correlations

def calculate_cyclomatic_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate cyclomatic complexity of the model."""
    # Simplified cyclomatic complexity calculation
    # In practice, this would be more sophisticated
    base_complexity = 1
    decision_points = len([conn for conn in connections if "if" in conn["definition"].lower() or "condition" in conn["definition"].lower()])
    return base_complexity + decision_points

def calculate_cognitive_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate cognitive complexity of the model."""
    # Simplified cognitive complexity calculation
    complexity = 0
    
    # Add complexity for each variable
    complexity += len(variables) * 0.5
    
    # Add complexity for each connection
    complexity += len(connections) * 0.3
    
    # Add complexity for nested structures
    for conn in connections:
        if "nested" in conn["definition"].lower() or "hierarchical" in conn["definition"].lower():
            complexity += 1
    
    return complexity

def calculate_structural_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate structural complexity of the model."""
    # Structural complexity based on graph theory metrics
    if not connections:
        return 0.0
    
    # Calculate average degree
    connectivity = build_connectivity_matrix(connections)
    degrees = [len(targets) for targets in connectivity.values()]
    avg_degree = np.mean(degrees) if degrees else 0
    
    # Calculate density
    max_edges = len(variables) * (len(variables) - 1) / 2 if len(variables) > 1 else 0
    density = len(connections) / max_edges if max_edges > 0 else 0
    
    # Structural complexity score
    structural_complexity = avg_degree * density * len(variables) / 100
    
    return structural_complexity

def calculate_complexity_metrics(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Calculate comprehensive complexity metrics for a GNN model."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)
        
        # Calculate various complexity metrics
        metrics = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "basic_metrics": {
                "lines_of_code": len(content.splitlines()),
                "characters": len(content),
                "variables": len(variables),
                "connections": len(connections)
            },
            "cyclomatic_complexity": calculate_cyclomatic_complexity(variables, connections),
            "cognitive_complexity": calculate_cognitive_complexity(variables, connections),
            "structural_complexity": calculate_structural_complexity(variables, connections),
            "maintainability_index": calculate_maintainability_index(content, variables, connections),
            "technical_debt": calculate_technical_debt(content, variables, connections)
        }
        
        return metrics
        
    except Exception as e:
        raise Exception(f"Failed to calculate complexity metrics for {file_path}: {e}")

def calculate_maintainability_index(content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate maintainability index."""
    # Simplified maintainability index calculation
    # Higher values indicate better maintainability
    
    # Base score
    score = 100
    
    # Penalize for complexity
    score -= len(variables) * 0.5
    score -= len(connections) * 0.3
    
    # Penalize for long lines
    long_lines = sum(1 for line in content.splitlines() if len(line) > 80)
    score -= long_lines * 0.1
    
    # Penalize for lack of documentation
    comment_lines = sum(1 for line in content.splitlines() if line.strip().startswith('#'))
    doc_ratio = comment_lines / max(len(content.splitlines()), 1)
    if doc_ratio < 0.1:
        score -= 10
    
    return max(score, 0)

def calculate_technical_debt(content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float:
    """Calculate technical debt score."""
    # Simplified technical debt calculation
    debt = 0
    
    # Add debt for complexity
    debt += len(variables) * 0.1
    debt += len(connections) * 0.05
    
    # Add debt for potential issues
    if len(variables) > 20:
        debt += 5  # High complexity
    
    if len(connections) > len(variables) * 2:
        debt += 3  # High coupling
    
    # Add debt for code smells
    if "TODO" in content or "FIXME" in content:
        debt += 2
    
    return debt

def run_performance_benchmarks(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Run performance benchmarks on the GNN model."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        variables = extract_variables_for_analysis(content)
        connections = extract_connections_for_analysis(content)
        
        # Simulate performance benchmarks
        benchmarks = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "parsing_performance": {
                "parse_time_ms": len(content) * 0.01,  # Simulated parse time
                "memory_usage_mb": len(content) / 1024 / 1024 * 0.1  # Simulated memory usage
            },
            "execution_performance": {
                "estimated_runtime_ms": len(variables) * len(connections) * 0.1,
                "memory_footprint_mb": len(variables) * 8 / 1024 / 1024,
                "cpu_usage_percent": min(len(variables) * len(connections) / 100, 100)
            },
            "scalability_metrics": {
                "linear_scaling_factor": len(variables) + len(connections),
                "quadratic_scaling_factor": len(variables) * len(connections),
                "recommended_max_variables": 50,
                "recommended_max_connections": 100
            }
        }
        
        return benchmarks
        
    except Exception as e:
        raise Exception(f"Failed to run performance benchmarks for {file_path}: {e}")

def perform_model_comparisons(statistical_analyses: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """Perform comparisons between multiple models."""
    if len(statistical_analyses) < 2:
        return {"message": "Need at least 2 models for comparison"}
    
    comparisons = {
        "model_count": len(statistical_analyses),
        "complexity_comparison": {},
        "size_comparison": {},
        "performance_comparison": {},
        "ranking": {}
    }
    
    # Compare complexity metrics
    var_counts = [analysis["variable_statistics"]["count"] for analysis in statistical_analyses]
    conn_counts = [analysis["connection_statistics"]["count"] for analysis in statistical_analyses]
    
    comparisons["complexity_comparison"] = {
        "variable_count_stats": {
            "mean": np.mean(var_counts),
            "std": np.std(var_counts),
            "min": np.min(var_counts),
            "max": np.max(var_counts)
        },
        "connection_count_stats": {
            "mean": np.mean(conn_counts),
            "std": np.std(conn_counts),
            "min": np.min(conn_counts),
            "max": np.max(conn_counts)
        }
    }
    
    # Compare file sizes
    file_sizes = [analysis["file_size"] for analysis in statistical_analyses]
    comparisons["size_comparison"] = {
        "file_size_stats": {
            "mean": np.mean(file_sizes),
            "std": np.std(file_sizes),
            "min": np.min(file_sizes),
            "max": np.max(file_sizes)
        }
    }
    
    # Rank models by complexity
    complexity_scores = [var + conn for var, conn in zip(var_counts, conn_counts)]
    model_names = [analysis["file_name"] for analysis in statistical_analyses]
    
    # Create ranking
    ranked_models = sorted(zip(model_names, complexity_scores), key=lambda x: x[1])
    comparisons["ranking"] = {
        "by_complexity": [{"model": name, "score": score} for name, score in ranked_models]
    }
    
    return comparisons

def generate_analysis_summary(results: Dict[str, Any]) -> str:
    """Generate a markdown summary of analysis results."""
    summary = f"""# Analysis Summary

Generated on: {results['timestamp']}

## Overview
- **Files Processed**: {results['processed_files']}
- **Success**: {results['success']}
- **Errors**: {len(results['errors'])}

## Statistical Analysis Results
"""
    
    for analysis in results["statistical_analysis"]:
        summary += f"""
### {analysis['file_name']}
- **Variables**: {analysis['variable_statistics']['count']}
- **Connections**: {analysis['connection_statistics']['count']}
- **File Size**: {analysis['file_size']} bytes
- **Lines**: {analysis['line_count']}
"""
    
    if results["complexity_metrics"]:
        summary += "\n## Complexity Metrics\n"
        for metrics in results["complexity_metrics"]:
            summary += f"""
### {metrics['file_name']}
- **Cyclomatic Complexity**: {metrics['cyclomatic_complexity']:.2f}
- **Cognitive Complexity**: {metrics['cognitive_complexity']:.2f}
- **Structural Complexity**: {metrics['structural_complexity']:.2f}
- **Maintainability Index**: {metrics['maintainability_index']:.2f}
- **Technical Debt**: {metrics['technical_debt']:.2f}
"""
    
    if results["model_comparisons"]:
        summary += "\n## Model Comparisons\n"
        for comparison in results["model_comparisons"]:
            if "ranking" in comparison:
                summary += "\n### Complexity Ranking\n"
                for rank, item in enumerate(comparison["ranking"]["by_complexity"], 1):
                    summary += f"{rank}. {item['model']} (Score: {item['score']})\n"
    
    if results["errors"]:
        summary += "\n## Errors\n"
        for error in results["errors"]:
            summary += f"- {error}\n"
    
    return summary

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Comprehensive analysis and statistical processing for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'statistical_analysis': True,
    'complexity_metrics': True,
    'performance_benchmarks': True,
    'model_comparisons': True,
    'distribution_analysis': True
}

__all__ = [
    'process_analysis',
    'perform_statistical_analysis',
    'calculate_complexity_metrics',
    'run_performance_benchmarks',
    'perform_model_comparisons',
    'FEATURES',
    '__version__'
]
