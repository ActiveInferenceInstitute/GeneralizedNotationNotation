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

# Phase 1.2: soft-import analyzer and post_simulation. Step 16 is not a
# hard-import step, so missing optional deps (pymdp/jax/pytorch analyzers,
# matplotlib, etc.) must degrade gracefully rather than crash at package load.
# When imports fail, _unavailable() returns a callable stub so downstream code
# can call any of these names without AttributeError.
import logging as _logging

_ANALYSIS_IMPORT_ERROR: str | None = None


def _unavailable(name: str):
    """Return a callable stub that logs a warning and returns None.

    Used for every symbol we couldn't import. Keeps the module's public API
    surface intact for __all__ while signaling degradation via logs.
    """
    def _stub(*_args, **_kwargs):  # pragma: no cover - exercised via test_analysis_soft_import
        _logging.getLogger("analysis").warning(
            f"analysis.{name} unavailable (analyzer import failed): {_ANALYSIS_IMPORT_ERROR}"
        )
        return None
    _stub.__name__ = name
    return _stub


try:
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
    _ANALYZER_IMPORTED = True
except ImportError as _err:  # pragma: no cover - exercised via test_analysis_soft_import
    _ANALYSIS_IMPORT_ERROR = str(_err)
    _ANALYZER_IMPORTED = False
    for _n in [
        "analyze_distributions", "analyze_framework_outputs", "build_connectivity_matrix",
        "calculate_cognitive_complexity", "calculate_complexity_metrics",
        "calculate_connection_statistics", "calculate_correlations",
        "calculate_cyclomatic_complexity", "calculate_maintainability_index",
        "calculate_section_statistics", "calculate_structural_complexity",
        "calculate_technical_debt", "calculate_variable_statistics",
        "count_type_distribution", "extract_connections", "extract_sections",
        "extract_variables", "generate_analysis_summary",
        "generate_framework_comparison_report", "perform_model_comparisons",
        "perform_statistical_analysis", "run_performance_benchmarks",
        "visualize_cross_framework_metrics",
    ]:
        globals()[_n] = _unavailable(_n)

try:
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
    _POST_SIM_IMPORTED = True
except ImportError as _err:  # pragma: no cover
    if _ANALYSIS_IMPORT_ERROR is None:
        _ANALYSIS_IMPORT_ERROR = str(_err)
    _POST_SIM_IMPORTED = False
    for _n in [
        "analyze_active_inference_metrics", "analyze_execution_results",
        "analyze_free_energy", "analyze_policy_convergence",
        "analyze_simulation_traces", "analyze_state_distributions",
        "animate_belief_evolution", "compare_framework_results",
        "compute_expected_free_energy", "compute_information_gain",
        "compute_kl_divergence", "compute_shannon_entropy",
        "compute_variational_free_energy", "extract_activeinference_jl_data",
        "extract_discopy_data", "extract_jax_data", "extract_pymdp_data",
        "extract_rxinfer_data", "generate_action_analysis",
        "generate_belief_heatmaps", "generate_cross_framework_comparison",
        "generate_free_energy_plots", "generate_observation_analysis",
        "plot_belief_evolution", "visualize_all_framework_outputs",
    ]:
        globals()[_n] = _unavailable(_n)

try:
    from .processor import convert_numpy_types, process_analysis
    _PROCESSOR_IMPORTED = True
except ImportError as _err:  # pragma: no cover
    if _ANALYSIS_IMPORT_ERROR is None:
        _ANALYSIS_IMPORT_ERROR = str(_err)
    _PROCESSOR_IMPORTED = False

    def convert_numpy_types(*_a, **_k):  # type: ignore[misc]
        return None

    def process_analysis(*_a, **_k) -> int:  # type: ignore[misc]
        _logging.getLogger("analysis").warning(
            f"analysis.process_analysis unavailable: {_ANALYSIS_IMPORT_ERROR}"
        )
        return 2

# Reflect load success in FEATURES so callers can probe availability.
FEATURES["analyzer_available"] = _ANALYZER_IMPORTED
FEATURES["post_simulation_available"] = _POST_SIM_IMPORTED
FEATURES["processor_available"] = _PROCESSOR_IMPORTED

# Note: framework-specific analyzers live in ``src/analysis/<framework>/analyzer.py``
# and are discovered by ``processor.process_analysis`` via ``importlib`` — no
# need to re-export per-framework aliases at the package level.

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
