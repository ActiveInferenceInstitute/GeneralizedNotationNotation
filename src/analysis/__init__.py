"""
Analysis module for GNN Processing Pipeline.

This module provides comprehensive analysis and statistical processing for GNN models.
"""

from typing import Any, Dict

__version__ = "1.6.0"
FEATURES = {
    "statistical_analysis": True,
    "framework_comparison": True,
    "post_simulation_analysis": True,
    "belief_visualization": True,
    "cross_framework_metrics": True,
    "pytorch_analysis": True,
    "numpyro_analysis": True,
    "mcp_integration": True
}

from .analyzer import (
    analyze_distributions,
    analyze_framework_outputs,
    build_connectivity_matrix,
    calculate_cognitive_complexity,
    calculate_complexity_metrics,
    calculate_connection_statistics,
    calculate_correlations,
    calculate_cyclomatic_complexity,
    calculate_maintainability_index,
    calculate_section_statistics,
    calculate_structural_complexity,
    calculate_technical_debt,
    calculate_variable_statistics,
    count_type_distribution,
    extract_connections,
    extract_sections,
    extract_variables,
    generate_analysis_summary,
    generate_framework_comparison_report,
    perform_model_comparisons,
    perform_statistical_analysis,
    run_performance_benchmarks,
    visualize_cross_framework_metrics,
)
from .post_simulation import (
    analyze_active_inference_metrics,
    analyze_execution_results,
    analyze_free_energy,
    analyze_policy_convergence,
    analyze_simulation_traces,
    analyze_state_distributions,
    animate_belief_evolution,
    compare_framework_results,
    compute_expected_free_energy,
    compute_information_gain,
    compute_kl_divergence,
    # Active Inference-specific statistical methods
    compute_shannon_entropy,
    compute_variational_free_energy,
    extract_activeinference_jl_data,
    extract_discopy_data,
    extract_jax_data,
    extract_pymdp_data,
    extract_rxinfer_data,
    generate_action_analysis,
    generate_belief_heatmaps,
    generate_cross_framework_comparison,
    generate_free_energy_plots,
    generate_observation_analysis,
    plot_belief_evolution,
    # Comprehensive visualization functions
    visualize_all_framework_outputs,
)
from .processor import convert_numpy_types, process_analysis

# Optional framework-specific analyzers (graceful import)
try:
    from .pytorch.analyzer import (
        generate_analysis_from_logs as pytorch_generate_analysis,
    )
except ImportError:
    pytorch_generate_analysis = None

try:
    from .numpyro.analyzer import (
        generate_analysis_from_logs as numpyro_generate_analysis,
    )
except ImportError:
    numpyro_generate_analysis = None


# Note: process_analysis is imported from processor.py at the top of this file.
# Do NOT redefine it here - that would shadow the comprehensive implementation.

def check_analysis_tools() -> Dict[str, Dict[str, Any]]:
    """Check availability of analysis tools."""
    import importlib
    tools = {}
    for pkg_name in ('numpy', 'pandas', 'scipy', 'matplotlib'):
        try:
            m = importlib.import_module(pkg_name)
            tools[pkg_name] = {'available': True, 'version': m.__version__}
        except ImportError:
            tools[pkg_name] = {'available': False, 'version': None}
    return tools


__all__ = [
    '__version__',
    'FEATURES',
    'process_analysis',
    'convert_numpy_types',
    'perform_statistical_analysis',
    'extract_variables',
    'extract_connections',
    'extract_sections',
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
    # Active Inference-specific statistical methods
    'compute_shannon_entropy',
    'compute_kl_divergence',
    'compute_variational_free_energy',
    'compute_expected_free_energy',
    'compute_information_gain',
    'analyze_active_inference_metrics',
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


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "analysis",
        "version": __version__,
        "description": "Statistical analysis and result aggregation for GNN pipeline outputs",
        "features": FEATURES,
    }
