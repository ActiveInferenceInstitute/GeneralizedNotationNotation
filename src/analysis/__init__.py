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
    generate_analysis_summary,
    analyze_framework_outputs,
    generate_framework_comparison_report,
    visualize_cross_framework_metrics
)
from .post_simulation import (
    analyze_simulation_traces,
    analyze_free_energy,
    analyze_policy_convergence,
    analyze_state_distributions,
    compare_framework_results,
    extract_pymdp_data,
    extract_rxinfer_data,
    extract_activeinference_jl_data,
    extract_jax_data,
    extract_discopy_data,
    analyze_execution_results,
    # New comprehensive visualization functions
    visualize_all_framework_outputs,
    generate_belief_heatmaps,
    generate_action_analysis,
    generate_free_energy_plots,
    generate_observation_analysis,
    generate_cross_framework_comparison,
    plot_belief_evolution,
    animate_belief_evolution
)


# Note: process_analysis is imported from processor.py at the top of this file.
# Do NOT redefine it here - that would shadow the comprehensive implementation.

def check_analysis_tools():
    """Check availability of analysis tools."""
    tools = {}
    
    # Check numpy
    try:
        import numpy
        tools['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        tools['numpy'] = {'available': False, 'version': None}
    
    # Check pandas
    try:
        import pandas
        tools['pandas'] = {
            'available': True,
            'version': pandas.__version__
        }
    except ImportError:
        tools['pandas'] = {'available': False, 'version': None}
    
    # Check scipy
    try:
        import scipy
        tools['scipy'] = {
            'available': True,
            'version': scipy.__version__
        }
    except ImportError:
        tools['scipy'] = {'available': False, 'version': None}
    
    # Check matplotlib
    try:
        import matplotlib
        tools['matplotlib'] = {
            'available': True,
            'version': matplotlib.__version__
        }
    except ImportError:
        tools['matplotlib'] = {'available': False, 'version': None}
    
    return tools


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
    'analyze_framework_outputs',
    'generate_framework_comparison_report',
    'visualize_cross_framework_metrics',
    # Post-simulation analysis functions
    'analyze_simulation_traces',
    'analyze_free_energy',
    'analyze_policy_convergence',
    'analyze_state_distributions',
    'compare_framework_results',
    'extract_pymdp_data',
    'extract_rxinfer_data',
    'extract_activeinference_jl_data',
    'extract_jax_data',
    'extract_discopy_data',
    'analyze_execution_results',
    # Comprehensive visualization functions
    'visualize_all_framework_outputs',
    'generate_belief_heatmaps',
    'generate_action_analysis',
    'generate_free_energy_plots',
    'generate_observation_analysis',
    'generate_cross_framework_comparison',
    'plot_belief_evolution',
    'animate_belief_evolution'
]
