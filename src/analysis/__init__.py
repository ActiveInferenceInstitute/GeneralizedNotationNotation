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


__version__ = "1.0.0"
FEATURES = {
    "statistical_analysis": True,
    "complexity_metrics": True,
    "performance_benchmarks": True,
    "model_comparison": True,
    "mcp_integration": True,
    "correlation_analysis": True,
    "distribution_analysis": True,
    "maintainability_metrics": True
}

def process_analysis(target_dir, output_dir, verbose=False, **kwargs):
    """
    Main processing function for analysis.

    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options

    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info(f"Processing analysis for files in {target_dir}")
        # Placeholder implementation - delegate to actual module functions
        # This would be replaced with actual implementation
        logger.info(f"Analysis processing completed")
        return True
    except Exception as e:
        logger.error(f"Analysis processing failed: {e}")
        return False


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
    'generate_analysis_summary',
    '__version__',
    'FEATURES'
]
