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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
                        logger.info(f"Reading PyMDP simulation data from {results_files[0].name}")
                        with open(results_files[0], 'r') as f:
                            file_data = json.load(f)
                            
                            # Extract from file data
                            if "beliefs" in file_data:
                                simulation_data["beliefs"] = file_data["beliefs"]
                                logger.debug(f"Extracted {len(file_data['beliefs'])} belief states")
                            if "actions" in file_data:
                                simulation_data["actions"] = file_data["actions"]
                                logger.debug(f"Extracted {len(file_data['actions'])} actions")
                            if "observations" in file_data:
                                simulation_data["observations"] = file_data["observations"]
                                logger.debug(f"Extracted {len(file_data['observations'])} observations")
                            
                            logger.info(f"Enhanced PyMDP data from {results_files[0].name}")
                    except Exception as e:
                        logger.warning(f"Failed to read simulation_results.json: {e}")
            
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
    Enhanced to read from collected files if available.
    """
    simulation_data = execution_result.get("simulation_data", {})
    
    # Try to read from collected files if available
    implementation_dir = execution_result.get("implementation_directory")
    if implementation_dir:
        try:
            impl_path = Path(implementation_dir)
            results_file = impl_path / "simulation_results.json"
            if results_file.exists():
                logger.info(f"Reading RxInfer simulation data from {results_file.name}")
                with open(results_file, 'r') as f:
                    file_data = json.load(f)
                    if "beliefs" in file_data:
                        simulation_data["beliefs"] = file_data["beliefs"]
                    if "true_states" in file_data:
                        simulation_data["true_states"] = file_data["true_states"]
                    if "observations" in file_data:
                        simulation_data["observations"] = file_data["observations"]
        except Exception as e:
            logger.debug(f"Error reading RxInfer files: {e}")

    return {
        "beliefs": simulation_data.get("beliefs", []),
        "true_states": simulation_data.get("true_states", []),
        "observations": simulation_data.get("observations", []),
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


def plot_belief_evolution(
    beliefs: List[List[float]],
    output_path: Path,
    title: str = "Belief Evolution",
    true_states: Optional[List[int]] = None
) -> str:
    """
    Plot belief evolution over time.
    """
    plt.figure(figsize=(10, 6))
    belief_array = np.array(beliefs)
    time_steps = range(len(beliefs))
    
    for i in range(belief_array.shape[1]):
        plt.plot(time_steps, belief_array[:, i], label=f"State {i+1}")
    
    if true_states:
        # Normalize true states if they are 1-indexed
        min_state = min(true_states)
        for t, s in enumerate(true_states):
            plt.scatter(t, 1.05, marker='*', color='black', alpha=0.5 if t > 0 else 0)
            plt.text(t, 1.1, f"S{s}", ha='center', fontsize=8)

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Probability")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    return str(output_path)

def animate_belief_evolution(
    beliefs: List[List[float]],
    output_path: Path,
    title: str = "Belief Evolution Animation"
) -> str:
    """
    Create a GIF animation of belief evolution.
    """
    belief_array = np.array(beliefs)
    n_steps, n_states = belief_array.shape
    
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot([], [], label=f"State {i+1}")[0] for i in range(n_states)]
    
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for i in range(n_states):
            lines[i].set_data(range(frame + 1), belief_array[:frame+1, i])
        return lines

    ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=5)
    ani.save(output_path, writer=writer)
    plt.close()
    return str(output_path)


def visualize_all_framework_outputs(
    execution_dir: Path,
    output_dir: Path,
    logger_instance: Optional[logging.Logger] = None
) -> List[str]:
    """
    Generate comprehensive visualizations for all raw execution outputs.
    
    This function creates a complete visualization suite for POMDP simulation outputs:
    - Belief trajectory plots per framework
    - Action selection histograms
    - Free energy evolution curves
    - State distribution heatmaps
    - Cross-framework metric comparisons
    
    Args:
        execution_dir: Directory containing execution results (e.g., output/12_execute_output)
        output_dir: Directory to save visualizations
        logger_instance: Optional logger instance
        
    Returns:
        List of generated visualization file paths
    """
    log = logger_instance or logger
    generated_files: List[str] = []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all execution data
    framework_data: Dict[str, Dict[str, Any]] = {}
    
    # Search for execution result files
    for result_file in execution_dir.rglob("*_results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            framework = data.get("framework", "unknown")
            model_name = data.get("model_name", result_file.parent.name)
            
            key = f"{framework}_{model_name}"
            if key not in framework_data:
                framework_data[key] = {
                    "framework": framework,
                    "model_name": model_name,
                    "results": []
                }
            framework_data[key]["results"].append(data)
            
        except Exception as e:
            log.warning(f"Failed to load {result_file}: {e}")
    
    # Also search for simulation_results.json files
    for sim_file in execution_dir.rglob("*simulation_results.json"):
        try:
            with open(sim_file, 'r') as f:
                data = json.load(f)
            
            # Determine framework from path
            path_parts = sim_file.parts
            framework = "unknown"
            for part in path_parts:
                if part in ["pymdp", "rxinfer", "activeinference_jl", "jax", "discopy"]:
                    framework = part
                    break
            
            model_name = sim_file.parent.parent.name if len(sim_file.parts) > 2 else "unknown"
            
            key = f"{framework}_{model_name}_sim"
            if key not in framework_data:
                framework_data[key] = {
                    "framework": framework,
                    "model_name": model_name,
                    "simulation_data": data
                }
            else:
                framework_data[key]["simulation_data"] = data
                
        except Exception as e:
            log.warning(f"Failed to load simulation file {sim_file}: {e}")
    
    if not framework_data:
        log.warning("No execution data found for visualization")
        return generated_files
    
    log.info(f"Found {len(framework_data)} framework/model combinations for visualization")
    
    # Generate visualizations for each framework/model
    for key, data in framework_data.items():
        framework = data["framework"]
        model_name = data["model_name"]
        
        try:
            # Extract simulation data
            sim_data = data.get("simulation_data", {})
            if not sim_data and data.get("results"):
                # Try to extract from first result
                result = data["results"][0]
                sim_data = result.get("simulation_data", {})
                
                # Also check implementation directory for files
                impl_dir = result.get("implementation_directory")
                if impl_dir:
                    impl_path = Path(impl_dir)
                    sim_data_dir = impl_path / "simulation_data"
                    if sim_data_dir.exists():
                        for json_file in sim_data_dir.glob("*.json"):
                            try:
                                with open(json_file, 'r') as f:
                                    file_data = json.load(f)
                                if isinstance(file_data, dict):
                                    sim_data.update(file_data)
                            except Exception:
                                pass
            
            # Generate belief heatmap
            beliefs = sim_data.get("beliefs", [])
            if beliefs and len(beliefs) > 1:
                heatmap_file = output_dir / f"{model_name}_{framework}_belief_heatmap.png"
                try:
                    generate_belief_heatmaps(beliefs, heatmap_file, f"Belief Evolution - {model_name} ({framework})")
                    generated_files.append(str(heatmap_file))
                    log.info(f"Generated belief heatmap: {heatmap_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate belief heatmap for {key}: {e}")
            
            # Generate action analysis
            actions = sim_data.get("actions", [])
            if actions:
                action_file = output_dir / f"{model_name}_{framework}_action_analysis.png"
                try:
                    generate_action_analysis(actions, action_file, f"Action Selection - {model_name} ({framework})")
                    generated_files.append(str(action_file))
                    log.info(f"Generated action analysis: {action_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate action analysis for {key}: {e}")
            
            # Generate free energy plot
            free_energy = sim_data.get("free_energy", [])
            if free_energy:
                fe_file = output_dir / f"{model_name}_{framework}_free_energy.png"
                try:
                    generate_free_energy_plots(free_energy, fe_file, f"Free Energy - {model_name} ({framework})")
                    generated_files.append(str(fe_file))
                    log.info(f"Generated free energy plot: {fe_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate free energy plot for {key}: {e}")
            
            # Generate observation analysis
            observations = sim_data.get("observations", [])
            if observations:
                obs_file = output_dir / f"{model_name}_{framework}_observations.png"
                try:
                    generate_observation_analysis(observations, obs_file, f"Observations - {model_name} ({framework})")
                    generated_files.append(str(obs_file))
                    log.info(f"Generated observation analysis: {obs_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate observation analysis for {key}: {e}")
                    
        except Exception as e:
            log.error(f"Failed to generate visualizations for {key}: {e}")
    
    # Generate cross-framework comparison if multiple frameworks
    frameworks = set(d["framework"] for d in framework_data.values())
    if len(frameworks) > 1:
        try:
            comparison_file = output_dir / "cross_framework_comparison.png"
            generate_cross_framework_comparison(framework_data, comparison_file)
            generated_files.append(str(comparison_file))
            log.info(f"Generated cross-framework comparison: {comparison_file.name}")
        except Exception as e:
            log.warning(f"Failed to generate cross-framework comparison: {e}")
    
    log.info(f"Generated {len(generated_files)} visualization files")
    return generated_files


def generate_belief_heatmaps(
    beliefs: List[List[float]],
    output_path: Path,
    title: str = "Belief State Evolution Heatmap"
) -> str:
    """
    Generate a heatmap visualization of belief state evolution over time.
    
    Args:
        beliefs: List of belief distributions at each timestep [[p1, p2, ...], ...]
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to the generated file
    """
    if not beliefs or len(beliefs) < 2:
        raise ValueError("Need at least 2 timesteps for heatmap")
    
    belief_array = np.array(beliefs)
    n_steps, n_states = belief_array.shape
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(belief_array.T, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("State")
    ax1.set_title(f"{title}\n(Heatmap)")
    ax1.set_yticks(range(n_states))
    ax1.set_yticklabels([f"S{i+1}" for i in range(n_states)])
    plt.colorbar(im, ax=ax1, label="Probability")
    
    # Line plot
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    for i in range(n_states):
        ax2.plot(range(n_steps), belief_array[:, i], label=f"State {i+1}", 
                 color=colors[i], linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"{title}\n(Trajectories)")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def generate_action_analysis(
    actions: List[int],
    output_path: Path,
    title: str = "Action Selection Analysis"
) -> str:
    """
    Generate visualization of action selection patterns.
    
    Args:
        actions: List of action indices taken at each timestep
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to the generated file
    """
    if not actions:
        raise ValueError("No actions provided")
    
    actions_array = np.array(actions)
    unique_actions = sorted(set(actions))
    n_actions = len(unique_actions)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Action histogram
    ax1 = axes[0]
    action_counts = [np.sum(actions_array == a) for a in unique_actions]
    colors = plt.cm.Set2(np.linspace(0, 1, n_actions))
    bars = ax1.bar(unique_actions, action_counts, color=colors)
    ax1.set_xlabel("Action")
    ax1.set_ylabel("Count")
    ax1.set_title("Action Frequency Distribution")
    ax1.set_xticks(unique_actions)
    ax1.set_xticklabels([f"A{a}" for a in unique_actions])
    
    # Add count labels on bars
    for bar, count in zip(bars, action_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=10)
    
    # Action sequence plot
    ax2 = axes[1]
    ax2.plot(range(len(actions)), actions, 'o-', markersize=4, linewidth=1, alpha=0.7)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Action")
    ax2.set_title("Action Sequence Over Time")
    ax2.set_yticks(unique_actions)
    ax2.set_yticklabels([f"A{a}" for a in unique_actions])
    ax2.grid(True, alpha=0.3)
    
    # Action transition matrix
    ax3 = axes[2]
    if len(actions) > 1:
        transition_matrix = np.zeros((n_actions, n_actions))
        action_to_idx = {a: i for i, a in enumerate(unique_actions)}
        for i in range(len(actions) - 1):
            from_idx = action_to_idx[actions[i]]
            to_idx = action_to_idx[actions[i + 1]]
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                       where=row_sums != 0, 
                                       out=np.zeros_like(transition_matrix))
        
        im = ax3.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        ax3.set_xlabel("Next Action")
        ax3.set_ylabel("Current Action")
        ax3.set_title("Action Transition Probabilities")
        ax3.set_xticks(range(n_actions))
        ax3.set_yticks(range(n_actions))
        ax3.set_xticklabels([f"A{a}" for a in unique_actions])
        ax3.set_yticklabels([f"A{a}" for a in unique_actions])
        
        # Add text annotations
        for i in range(n_actions):
            for j in range(n_actions):
                text = ax3.text(j, i, f"{transition_matrix[i, j]:.2f}",
                               ha="center", va="center", color="black" if transition_matrix[i, j] < 0.5 else "white")
        
        plt.colorbar(im, ax=ax3, label="Probability")
    else:
        ax3.text(0.5, 0.5, "Need > 1 action\nfor transitions", 
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Action Transition Probabilities")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def generate_free_energy_plots(
    free_energy: List[float],
    output_path: Path,
    title: str = "Free Energy Dynamics"
) -> str:
    """
    Generate visualization of free energy evolution.
    
    Includes:
    - Free energy over time
    - Moving average trend
    - Convergence analysis
    
    Args:
        free_energy: List of free energy values over time
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to the generated file
    """
    if not free_energy:
        raise ValueError("No free energy values provided")
    
    fe_array = np.array(free_energy)
    n_steps = len(fe_array)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Main free energy plot
    ax1 = axes[0, 0]
    ax1.plot(range(n_steps), fe_array, 'b-', linewidth=1.5, label='Free Energy')
    
    # Add moving average if enough points
    if n_steps > 5:
        window = min(5, n_steps // 3)
        moving_avg = np.convolve(fe_array, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, n_steps), moving_avg, 'r--', linewidth=2, 
                 label=f'{window}-step Moving Average')
    
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Free Energy")
    ax1.set_title("Free Energy Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution of free energy values
    ax2 = axes[0, 1]
    ax2.hist(fe_array, bins=min(20, n_steps), color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(np.mean(fe_array), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(fe_array):.3f}')
    ax2.axvline(np.median(fe_array), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(fe_array):.3f}')
    ax2.set_xlabel("Free Energy")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Free Energy Distribution")
    ax2.legend()
    
    # Rate of change
    ax3 = axes[1, 0]
    if n_steps > 1:
        fe_diff = np.diff(fe_array)
        ax3.bar(range(len(fe_diff)), fe_diff, color=['green' if d < 0 else 'red' for d in fe_diff], alpha=0.7)
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("ΔFE")
        ax3.set_title("Free Energy Change per Step")
        
        # Add summary statistics
        positive_changes = np.sum(fe_diff > 0)
        negative_changes = np.sum(fe_diff < 0)
        ax3.text(0.02, 0.98, f"↑ Increases: {positive_changes}\n↓ Decreases: {negative_changes}", 
                 transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Convergence analysis
    ax4 = axes[1, 1]
    if n_steps > 10:
        # Calculate rolling variance
        window = max(3, n_steps // 10)
        rolling_var = []
        for i in range(window, n_steps + 1):
            rolling_var.append(np.var(fe_array[i-window:i]))
        
        ax4.plot(range(window, n_steps + 1), rolling_var, 'purple', linewidth=1.5)
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Rolling Variance")
        ax4.set_title(f"Convergence Analysis ({window}-step variance)")
        ax4.grid(True, alpha=0.3)
        
        # Determine convergence status
        if rolling_var:
            final_var = rolling_var[-1]
            converged = final_var < 0.1
            status = "✓ Converged" if converged else "⚠ Not Converged"
            ax4.text(0.98, 0.98, f"{status}\nFinal Variance: {final_var:.4f}", 
                     transform=ax4.transAxes, verticalalignment='top', horizontalalignment='right',
                     fontsize=10, bbox=dict(boxstyle='round', 
                                            facecolor='lightgreen' if converged else 'lightyellow', alpha=0.7))
    else:
        ax4.text(0.5, 0.5, "Need > 10 steps\nfor convergence analysis", 
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Convergence Analysis")
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def generate_observation_analysis(
    observations: List[int],
    output_path: Path,
    title: str = "Observation Analysis"
) -> str:
    """
    Generate visualization of observation patterns.
    
    Args:
        observations: List of observation indices
        output_path: Path to save the visualization
        title: Title for the plot
        
    Returns:
        Path to the generated file
    """
    if not observations:
        raise ValueError("No observations provided")
    
    obs_array = np.array(observations)
    unique_obs = sorted(set(observations))
    n_obs = len(unique_obs)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Observation frequency
    ax1 = axes[0]
    obs_counts = [np.sum(obs_array == o) for o in unique_obs]
    colors = plt.cm.Pastel1(np.linspace(0, 1, n_obs))
    ax1.bar(unique_obs, obs_counts, color=colors, edgecolor='black')
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("Count")
    ax1.set_title("Observation Frequency")
    ax1.set_xticks(unique_obs)
    ax1.set_xticklabels([f"O{o}" for o in unique_obs])
    
    # Observation sequence
    ax2 = axes[1]
    ax2.scatter(range(len(observations)), observations, c=observations, cmap='tab10', s=30, alpha=0.7)
    ax2.plot(range(len(observations)), observations, 'gray', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Observation")
    ax2.set_title("Observation Sequence")
    ax2.set_yticks(unique_obs)
    ax2.set_yticklabels([f"O{o}" for o in unique_obs])
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def generate_cross_framework_comparison(
    framework_data: Dict[str, Dict[str, Any]],
    output_path: Path
) -> str:
    """
    Generate cross-framework comparison visualization.
    
    Args:
        framework_data: Dictionary mapping framework keys to their data
        output_path: Path to save the visualization
        
    Returns:
        Path to the generated file
    """
    frameworks = []
    metrics = {"execution_time": [], "steps_completed": [], "success_rate": []}
    
    for key, data in framework_data.items():
        framework = data.get("framework", "unknown")
        frameworks.append(framework)
        
        results = data.get("results", [])
        if results:
            result = results[0]
            metrics["execution_time"].append(result.get("execution_time", 0))
            metrics["steps_completed"].append(result.get("steps_completed", 0))
            metrics["success_rate"].append(1 if result.get("success", False) else 0)
        else:
            for m in metrics:
                metrics[m].append(0)
    
    if not frameworks:
        raise ValueError("No framework data for comparison")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(frameworks)))
    
    # Execution time comparison
    ax1 = axes[0]
    bars = ax1.bar(range(len(frameworks)), metrics["execution_time"], color=colors)
    ax1.set_xlabel("Framework")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Execution Time Comparison")
    ax1.set_xticks(range(len(frameworks)))
    ax1.set_xticklabels(frameworks, rotation=45, ha='right')
    
    # Steps completed comparison
    ax2 = axes[1]
    ax2.bar(range(len(frameworks)), metrics["steps_completed"], color=colors)
    ax2.set_xlabel("Framework")
    ax2.set_ylabel("Steps Completed")
    ax2.set_title("Simulation Steps Comparison")
    ax2.set_xticks(range(len(frameworks)))
    ax2.set_xticklabels(frameworks, rotation=45, ha='right')
    
    # Success rate comparison
    ax3 = axes[2]
    success_colors = ['green' if s == 1 else 'red' for s in metrics["success_rate"]]
    ax3.bar(range(len(frameworks)), metrics["success_rate"], color=success_colors, alpha=0.7)
    ax3.set_xlabel("Framework")
    ax3.set_ylabel("Success (1=Yes, 0=No)")
    ax3.set_title("Execution Success Comparison")
    ax3.set_xticks(range(len(frameworks)))
    ax3.set_xticklabels(frameworks, rotation=45, ha='right')
    ax3.set_ylim(-0.1, 1.1)
    
    plt.suptitle("Cross-Framework Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


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
                                    # Try to read simulation data files directly
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

                # Validate serializability with logic similar to processor.py
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
                    # Test dump to catch circular references early
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
            # Prepare results for comparison
            comparison_input = {}
            for framework, results in framework_data.items():
                if results:
                    # Use first result as representative
                    comparison_input[framework] = results[0]
            
            comparison = compare_framework_results(comparison_input, model_name or "unknown")
            analysis_results["cross_framework_comparison"] = comparison
        
        # Trigger visualizations and animations for the model results
        if model_name:
            results_dir = execution_results_dir.parent / "analysis_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            for framework, data in analysis_results["framework_results"].items():
                for i, analysis in enumerate(data.get("analyses", [])):
                    # Check if we have belief data to plot/animate
                    if "beliefs" in analysis or (isinstance(analysis, dict) and "simulation_data" in analysis and "beliefs" in analysis["simulation_data"]):
                        beliefs = analysis.get("beliefs") or analysis["simulation_data"].get("beliefs")
                        if beliefs:
                            # Plot
                            plot_file = results_dir / f"{model_name}_{framework}_beliefs.png"
                            plot_belief_evolution(beliefs, plot_file, title=f"Beliefs - {model_name} ({framework})")
                            
                            # Animate
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

