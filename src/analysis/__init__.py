"""
Analysis module for GNN Processing Pipeline.

This module provides comprehensive analysis and statistical processing for GNN models.
"""

from .processor import process_analysis, convert_numpy_types
from .analyzer import (
    perform_statistical_analysis,
    extract_variables_for_analysis,
    extract_connections_for_analysis,
    extract_sections_for_analysis,
    calculate_variable_statistics,
    calculate_connection_statistics,
    calculate_section_statistics,
    count_type_distribution,
    build_connectivity_matrix,
    analyze_distributions,
    calculate_correlations,
    calculate_cyclomatic_complexity,
    calculate_cognitive_complexity,
    calculate_structural_complexity,
    calculate_complexity_metrics,
    calculate_maintainability_index,
    calculate_technical_debt,
    run_performance_benchmarks,
    perform_model_comparisons,
    generate_analysis_summary
)

__all__ = [
    'process_analysis',
    'convert_numpy_types',
    'perform_statistical_analysis',
    'extract_variables_for_analysis',
    'extract_connections_for_analysis',
    'extract_sections_for_analysis',
    'calculate_variable_statistics',
    'calculate_connection_statistics',
    'calculate_section_statistics',
    'count_type_distribution',
    'build_connectivity_matrix',
    'analyze_distributions',
    'calculate_correlations',
    'calculate_cyclomatic_complexity',
    'calculate_cognitive_complexity',
    'calculate_structural_complexity',
    'calculate_complexity_metrics',
    'calculate_maintainability_index',
    'calculate_technical_debt',
    'run_performance_benchmarks',
    'perform_model_comparisons',
    'generate_analysis_summary'
]
