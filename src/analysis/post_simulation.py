#!/usr/bin/env python3
"""
Post-Simulation Analysis Module

This module provides generic post-simulation analysis methods that work across
all frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy).

Implementation is split across sub-modules for maintainability:
- trace_analysis: Framework-agnostic trace, free energy, policy, state analysis
- framework_extractors: Framework-specific data extraction (extract_*_data)
- math_utils: Active Inference statistical functions (entropy, KL, VFE, EFE)
- visualizations: All plotting, animation, and dashboard generation

This file re-exports all public names for backward compatibility.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Re-export from trace_analysis
from .trace_analysis import (
    analyze_simulation_traces,
    analyze_free_energy,
    analyze_policy_convergence,
    analyze_state_distributions,
    compare_framework_results,
)

# Re-export from framework_extractors
from .framework_extractors import (
    extract_pymdp_data,
    extract_rxinfer_data,
    extract_activeinference_jl_data,
    extract_jax_data,
    extract_discopy_data,
)

# Re-export from math_utils
from .math_utils import (
    compute_shannon_entropy,
    compute_kl_divergence,
    compute_variational_free_energy,
    compute_expected_free_energy,
    compute_information_gain,
    analyze_active_inference_metrics,
)

# Re-export from visualizations
from .visualizations import (
    plot_belief_evolution,
    animate_belief_evolution,
    visualize_all_framework_outputs,
    generate_belief_heatmaps,
    generate_action_analysis,
    generate_free_energy_plots,
    generate_observation_analysis,
    generate_unified_framework_dashboard,
    generate_cross_framework_comparison,
)


def analyze_execution_results(
    execution_results_dir: Path,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze execution results from all frameworks.

    Args:
        execution_results_dir: Directory containing execution results
        model_name: Optional model name filter

    Returns:
        Dictionary with comprehensive analysis results
    """
    try:
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "execution_results_dir": str(execution_results_dir),
            "framework_results": {},
            "cross_framework_comparison": {}
        }

        # Find all result JSON files
        result_files = list(execution_results_dir.rglob("*_results.json"))

        if not result_files:
            logger.warning(f"No execution result files found in {execution_results_dir}")
            return analysis_results

        # Group by framework
        framework_data = {}

        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)

                framework = result_data.get("framework", "unknown")
                file_model_name = result_data.get("model_name", "unknown")

                # Filter by model name if specified
                if model_name and file_model_name != model_name:
                    continue

                if framework not in framework_data:
                    framework_data[framework] = []

                framework_data[framework].append(result_data)

            except Exception as e:
                logger.warning(f"Failed to load result file {result_file}: {e}")

        # Analyze each framework's results
        for framework, results in framework_data.items():
            try:
                framework_analysis = {
                    "framework": framework,
                    "result_count": len(results),
                    "analyses": []
                }

                for result in results:
                    # Extract framework-specific data
                    try:
                        if framework == "pymdp":
                            extracted = extract_pymdp_data(result)
                        elif framework == "rxinfer":
                            extracted = extract_rxinfer_data(result)
                        elif framework == "activeinference_jl":
                            extracted = extract_activeinference_jl_data(result)
                        elif framework == "jax":
                            extracted = extract_jax_data(result)
                        elif framework == "discopy":
                            extracted = extract_discopy_data(result)
                        else:
                            extracted = result.get("simulation_data", {}) or {}

                        # Also try to read from collected files if extraction didn't find data
                        if isinstance(extracted, dict) and not extracted.get("beliefs") and not extracted.get("observations"):
                            implementation_dir = result.get("implementation_directory")
                            if implementation_dir:
                                try:
                                    impl_path = Path(implementation_dir)
                                    sim_data_dir = impl_path / "simulation_data"
                                    if sim_data_dir.exists():
                                        results_files = list(sim_data_dir.glob("*.json"))
                                        for results_file in results_files:
                                            try:
                                                with open(results_file, 'r') as f:
                                                    file_data = json.load(f)
                                                    if isinstance(file_data, dict):
                                                        if "beliefs" in file_data and not extracted.get("beliefs"):
                                                            extracted["beliefs"] = file_data["beliefs"]
                                                        if "actions" in file_data and not extracted.get("actions"):
                                                            extracted["actions"] = file_data["actions"]
                                                        if "observations" in file_data and not extracted.get("observations"):
                                                            extracted["observations"] = file_data["observations"]
                                            except Exception:
                                                pass
                                except Exception as e:
                                    logger.debug(f"Error reading files for {framework}: {e}")

                        # Run generic analyses
                        model_name_for_analysis = result.get("model_name", "unknown")

                        if isinstance(extracted, dict):
                            if extracted.get("free_energy"):
                                fe_analysis = analyze_free_energy(
                                    extracted["free_energy"],
                                    framework,
                                    model_name_for_analysis
                                )
                                framework_analysis["analyses"].append(fe_analysis)

                            if extracted.get("traces"):
                                trace_analysis = analyze_simulation_traces(
                                    extracted["traces"],
                                    framework,
                                    model_name_for_analysis
                                )
                                framework_analysis["analyses"].append(trace_analysis)

                            if extracted.get("policy"):
                                policy_analysis = analyze_policy_convergence(
                                    extracted["policy"],
                                    framework,
                                    model_name_for_analysis
                                )
                                framework_analysis["analyses"].append(policy_analysis)
                    except Exception as e:
                        logger.warning(f"Error analyzing result for {framework}: {e}")
                        framework_analysis["analyses"].append({"error": str(e)})

                # Validate serializability
                def safe_json_default(obj):
                    if isinstance(obj, Path):
                        return str(obj)
                    if isinstance(obj, (np.integer, int)):
                        return int(obj)
                    if isinstance(obj, (np.floating, float)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (set, frozenset)):
                        return list(obj)
                    if hasattr(obj, '__dict__'):
                        return str(obj)
                    return str(obj)

                try:
                    json.dumps(framework_analysis, default=safe_json_default)
                    analysis_results["framework_results"][framework] = framework_analysis
                except Exception as e:
                    logger.error(f"Circular reference or serialization error in {framework} analysis: {e}")
                    analysis_results["framework_results"][framework] = {
                        "framework": framework,
                        "error": f"Serialization failed: {e}"
                    }

            except Exception as e:
                logger.error(f"Failed to analyze framework {framework}: {e}")
                analysis_results["framework_results"][framework] = {"error": str(e)}

        # Cross-framework comparison
        if len(framework_data) > 1:
            comparison_input = {}
            for framework, results in framework_data.items():
                if results:
                    comparison_input[framework] = results[0]

            comparison = compare_framework_results(comparison_input, model_name or "unknown")
            analysis_results["cross_framework_comparison"] = comparison

        # Trigger visualizations and animations for the model results
        if model_name:
            results_dir = execution_results_dir.parent / "analysis_results"
            results_dir.mkdir(parents=True, exist_ok=True)

            for framework, data in analysis_results["framework_results"].items():
                for i, analysis in enumerate(data.get("analyses", [])):
                    if "beliefs" in analysis or (isinstance(analysis, dict) and "simulation_data" in analysis and "beliefs" in analysis["simulation_data"]):
                        beliefs = analysis.get("beliefs") or analysis["simulation_data"].get("beliefs")
                        if beliefs:
                            plot_file = results_dir / f"{model_name}_{framework}_beliefs.png"
                            plot_belief_evolution(beliefs, plot_file, title=f"Beliefs - {model_name} ({framework})")

                            anim_file = results_dir / f"{model_name}_{framework}_beliefs.gif"
                            try:
                                animate_belief_evolution(beliefs, anim_file, title=f"Evolution - {model_name} ({framework})")
                            except Exception as e:
                                logger.warning(f"Animation failed for {framework}: {e}")

                            analysis["plots"] = analysis.get("plots", [])
                            analysis["plots"].extend([str(plot_file), str(anim_file)])

        return analysis_results

    except Exception as e:
        logger.error(f"Error analyzing execution results: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
