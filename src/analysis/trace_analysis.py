#!/usr/bin/env python3
"""
Trace Analysis Sub-module

Framework-agnostic analysis of simulation traces, free energy dynamics,
policy convergence, state distributions, and cross-framework comparison.

Extracted from post_simulation.py for maintainability.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def analyze_simulation_traces(
    traces: List[Any],
    framework: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Analyze simulation traces (state/observation/action trajectories).

    Args:
        traces: List of trace data (format depends on framework)
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with trace analysis results
    """
    try:
        analysis = {
            "framework": framework,
            "model_name": model_name,
            "trace_count": len(traces),
            "trace_lengths": [],
            "state_entropy": [],
            "observation_diversity": [],
            "action_distribution": {},
            "convergence_metrics": {}
        }

        if not traces:
            return analysis

        # Extract trace lengths
        for trace in traces:
            if isinstance(trace, (list, tuple)):
                analysis["trace_lengths"].append(len(trace))
            elif isinstance(trace, dict):
                analysis["trace_lengths"].append(len(trace.get("states", [])))

        # Calculate statistics
        if analysis["trace_lengths"]:
            analysis["avg_trace_length"] = np.mean(analysis["trace_lengths"])
            analysis["max_trace_length"] = np.max(analysis["trace_lengths"])
            analysis["min_trace_length"] = np.min(analysis["trace_lengths"])

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing simulation traces: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "error": str(e)
        }


def analyze_free_energy(
    free_energy_values: List[float],
    framework: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Analyze free energy dynamics.

    Args:
        free_energy_values: List of free energy values over time
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with free energy analysis results
    """
    try:
        analysis = {
            "framework": framework,
            "model_name": model_name,
            "free_energy_count": len(free_energy_values),
            "free_energy_values": free_energy_values
        }

        if not free_energy_values:
            return analysis

        fe_array = np.array(free_energy_values)

        # Calculate statistics
        analysis["mean_free_energy"] = float(np.mean(fe_array))
        analysis["std_free_energy"] = float(np.std(fe_array))
        analysis["min_free_energy"] = float(np.min(fe_array))
        analysis["max_free_energy"] = float(np.max(fe_array))

        # Calculate trend (decreasing = good for Active Inference)
        if len(fe_array) > 1:
            trend = np.polyfit(range(len(fe_array)), fe_array, 1)[0]
            analysis["free_energy_trend"] = float(trend)
            analysis["free_energy_decreasing"] = trend < 0

        # Calculate convergence (variance in last 20% of values)
        if len(fe_array) > 5:
            last_portion = fe_array[int(0.8 * len(fe_array)):]
            analysis["convergence_variance"] = float(np.var(last_portion))
            analysis["converged"] = analysis["convergence_variance"] < 0.1

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing free energy: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "error": str(e)
        }


def analyze_policy_convergence(
    policy_traces: List[Any],
    framework: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Analyze policy evolution and convergence.

    Args:
        policy_traces: List of policy distributions over time
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with policy convergence analysis
    """
    try:
        analysis = {
            "framework": framework,
            "model_name": model_name,
            "policy_count": len(policy_traces),
            "policy_entropy": [],
            "policy_stability": {}
        }

        if not policy_traces:
            return analysis

        # Calculate entropy for each policy
        for policy in policy_traces:
            if isinstance(policy, (list, tuple, np.ndarray)):
                policy_array = np.array(policy)
                # Normalize to probabilities
                policy_array = policy_array / np.sum(policy_array) if np.sum(policy_array) > 0 else policy_array
                # Calculate entropy
                entropy = -np.sum(policy_array * np.log(policy_array + 1e-10))
                analysis["policy_entropy"].append(float(entropy))

        # Calculate stability (variance in policy entropy)
        if analysis["policy_entropy"]:
            analysis["policy_stability"]["entropy_mean"] = float(np.mean(analysis["policy_entropy"]))
            analysis["policy_stability"]["entropy_std"] = float(np.std(analysis["policy_entropy"]))
            analysis["policy_stability"]["stable"] = analysis["policy_stability"]["entropy_std"] < 0.1

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing policy convergence: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "error": str(e)
        }


def analyze_state_distributions(
    state_traces: List[Any],
    framework: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Analyze belief state distributions.

    Args:
        state_traces: List of state distributions over time
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with state distribution analysis
    """
    try:
        analysis = {
            "framework": framework,
            "model_name": model_name,
            "state_count": len(state_traces),
            "state_entropy": [],
            "state_diversity": {}
        }

        if not state_traces:
            return analysis

        # Calculate entropy for each state distribution
        for state in state_traces:
            if isinstance(state, (list, tuple, np.ndarray)):
                state_array = np.array(state)
                # Normalize to probabilities
                state_array = state_array / np.sum(state_array) if np.sum(state_array) > 0 else state_array
                # Calculate entropy
                entropy = -np.sum(state_array * np.log(state_array + 1e-10))
                analysis["state_entropy"].append(float(entropy))

        # Calculate diversity metrics
        if analysis["state_entropy"]:
            analysis["state_diversity"]["mean_entropy"] = float(np.mean(analysis["state_entropy"]))
            analysis["state_diversity"]["std_entropy"] = float(np.std(analysis["state_entropy"]))

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing state distributions: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "error": str(e)
        }


def compare_framework_results(
    framework_results: Dict[str, Dict[str, Any]],
    model_name: str
) -> Dict[str, Any]:
    """
    Compare results across different frameworks.

    Args:
        framework_results: Dictionary mapping framework names to their results
        model_name: Model name

    Returns:
        Dictionary with cross-framework comparison
    """
    try:
        comparison = {
            "model_name": model_name,
            "frameworks_compared": list(framework_results.keys()),
            "framework_count": len(framework_results),
            "comparisons": {}
        }

        if len(framework_results) < 2:
            comparison["message"] = "Need at least 2 frameworks for comparison"
            return comparison

        # Compare free energy if available
        fe_comparison = {}
        for framework, results in framework_results.items():
            if "free_energy" in results.get("simulation_data", {}):
                fe_values = results["simulation_data"]["free_energy"]
                if fe_values:
                    fe_comparison[framework] = {
                        "mean": float(np.mean(fe_values)),
                        "min": float(np.min(fe_values)),
                        "max": float(np.max(fe_values))
                    }

        if fe_comparison:
            comparison["comparisons"]["free_energy"] = fe_comparison
            # Find best (lowest mean free energy)
            best_framework = min(fe_comparison.items(), key=lambda x: x[1]["mean"])
            comparison["comparisons"]["best_free_energy"] = {
                "framework": best_framework[0],
                "mean_fe": best_framework[1]["mean"]
            }

        # Compare execution times
        exec_times = {}
        for framework, results in framework_results.items():
            if "execution_time" in results:
                exec_times[framework] = results["execution_time"]

        if exec_times:
            comparison["comparisons"]["execution_time"] = exec_times
            fastest_framework = min(exec_times.items(), key=lambda x: x[1])
            comparison["comparisons"]["fastest_execution"] = {
                "framework": fastest_framework[0],
                "time": fastest_framework[1]
            }

        # Compare success rates
        success_rates = {}
        for framework, results in framework_results.items():
            success_rates[framework] = results.get("success", False)

        if success_rates:
            comparison["comparisons"]["success_rates"] = success_rates

        return comparison

    except Exception as e:
        logger.error(f"Error comparing framework results: {e}")
        return {
            "model_name": model_name,
            "error": str(e)
        }
