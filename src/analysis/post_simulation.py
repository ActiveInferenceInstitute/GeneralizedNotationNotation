#!/usr/bin/env python3
"""
Post-Simulation Analysis Module

This module provides generic post-simulation analysis methods that work across
all frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy).

Features:
- Framework-agnostic analysis of simulation traces
- Free energy dynamics analysis
- Policy convergence analysis
- State distribution analysis
- Cross-framework comparison
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
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


def extract_pymdp_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PyMDP-specific data from execution result.
    Enhanced to read from collected files if available.
    
    Args:
        execution_result: Execution result dictionary
        
    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})
    
    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)
            
            # Read simulation_results.json if available
            sim_data_dir = impl_path / "simulation_data"
            if sim_data_dir.exists():
                results_files = list(sim_data_dir.glob("*simulation_results.json"))
                if results_files:
                    try:
                        with open(results_files[0], 'r') as f:
                            file_data = json.load(f)
                            
                            # Extract from file data
                            if "beliefs" in file_data:
                                simulation_data["beliefs"] = file_data["beliefs"]
                            if "actions" in file_data:
                                simulation_data["actions"] = file_data["actions"]
                            if "observations" in file_data:
                                simulation_data["observations"] = file_data["observations"]
                            
                            logger.info(f"Enhanced PyMDP data from {results_files[0].name}")
                    except Exception as e:
                        logger.debug(f"Failed to read simulation_results.json: {e}")
            
            # Count visualizations
            viz_dir = impl_path / "visualizations"
            if viz_dir.exists():
                viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.svg"))
                if viz_files:
                    simulation_data["visualization_count"] = len(viz_files)
                    simulation_data["visualization_files"] = [str(f.name) for f in viz_files]
                    
        except Exception as e:
            logger.debug(f"Error reading PyMDP files: {e}")
    
    return {
        "traces": simulation_data.get("traces", []),
        "free_energy": simulation_data.get("free_energy", []),
        "states": simulation_data.get("states", []),
        "observations": simulation_data.get("observations", []),
        "actions": simulation_data.get("actions", []),
        "policy": simulation_data.get("policy", []),
        "beliefs": simulation_data.get("beliefs", []),
        "visualization_count": simulation_data.get("visualization_count", 0),
        "visualization_files": simulation_data.get("visualization_files", [])
    }


def extract_rxinfer_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract RxInfer.jl-specific data from execution result.
    
    Args:
        execution_result: Execution result dictionary
        
    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})
    
    return {
        "posterior": simulation_data.get("posterior", []),
        "inference_data": simulation_data.get("inference_data", [])
    }


def extract_activeinference_jl_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ActiveInference.jl-specific data from execution result.
    
    Args:
        execution_result: Execution result dictionary
        
    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})
    
    return {
        "free_energy": simulation_data.get("free_energy", []),
        "beliefs": simulation_data.get("beliefs", []),
        "states": simulation_data.get("states", [])
    }


def extract_jax_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract JAX-specific data from execution result.
    
    Args:
        execution_result: Execution result dictionary
        
    Returns:
        Extracted simulation data
    """
    # Similar to PyMDP
    return extract_pymdp_data(execution_result)


def extract_discopy_data(execution_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract DisCoPy-specific data from execution result.
    
    Args:
        execution_result: Execution result dictionary
        
    Returns:
        Extracted simulation data
    """
    simulation_data = execution_result.get("simulation_data", {})
    
    return {
        "diagrams": simulation_data.get("diagrams", []),
        "circuits": simulation_data.get("circuits", [])
    }


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
            framework_analysis = {
                "framework": framework,
                "result_count": len(results),
                "analyses": []
            }
            
            for result in results:
                # Extract framework-specific data
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
                    extracted = result.get("simulation_data", {})
                
                # Also try to read from collected files if extraction didn't find data
                if not extracted.get("beliefs") and not extracted.get("observations"):
                    implementation_dir = result.get("implementation_directory")
                    if implementation_dir:
                        try:
                            impl_path = Path(implementation_dir)
                            # Try to read simulation data files directly
                            sim_data_dir = impl_path / "simulation_data"
                            if sim_data_dir.exists():
                                results_files = list(sim_data_dir.glob("*.json"))
                                for results_file in results_files:
                                    try:
                                        with open(results_file, 'r') as f:
                                            file_data = json.load(f)
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
            
            analysis_results["framework_results"][framework] = framework_analysis
        
        # Cross-framework comparison
        if len(framework_data) > 1:
            # Prepare results for comparison
            comparison_input = {}
            for framework, results in framework_data.items():
                if results:
                    # Use first result as representative
                    comparison_input[framework] = results[0]
            
            comparison = compare_framework_results(comparison_input, model_name or "unknown")
            analysis_results["cross_framework_comparison"] = comparison
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing execution results: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

