"""
Visualization functions for post-simulation analysis.

Provides plot_belief_evolution, animate_belief_evolution, visualize_all_framework_outputs,
generate_belief_heatmaps, generate_action_analysis, generate_free_energy_plots,
generate_observation_analysis, generate_unified_framework_dashboard,
generate_cross_framework_comparison, and analyze_execution_results.

Extracted from post_simulation.py for maintainability.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .trace_analysis import (
    analyze_simulation_traces,
    analyze_free_energy,
    analyze_policy_convergence,
    compare_framework_results,
)
from .framework_extractors import (
    extract_pymdp_data,
    extract_rxinfer_data,
    extract_activeinference_jl_data,
    extract_jax_data,
    extract_discopy_data,
)
from .math_utils import compute_shannon_entropy

logger = logging.getLogger(__name__)


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


def _normalize_framework_name(framework: str) -> str:
    """
    Normalize framework names to canonical form.

    Consolidates variants like PyMDP, pymdp, pymdp_gen -> pymdp
    """
    if not framework:
        return "unknown"

    fw_lower = framework.lower()

    # Consolidate pymdp variants
    if fw_lower in ["pymdp", "pymdp_gen"] or fw_lower.startswith("pymdp"):
        return "pymdp"

    # Consolidate rxinfer variants
    if fw_lower in ["rxinfer", "rxinfer_jl"]:
        return "rxinfer"

    # Consolidate activeinference variants
    if fw_lower in ["activeinference_jl", "activeinference"]:
        return "activeinference_jl"

    return fw_lower


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

            # Normalize framework name to canonical form
            raw_framework = data.get("framework", "unknown")
            framework = _normalize_framework_name(raw_framework)
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

    # Also search for simulation_results.json files - MERGE into existing keys
    for sim_file in execution_dir.rglob("*simulation_results.json"):
        try:
            with open(sim_file, 'r') as f:
                data = json.load(f)

            # Determine framework from path or file content
            path_parts = sim_file.parts
            framework = "unknown"
            for part in path_parts:
                if part in ["pymdp", "pymdp_gen", "rxinfer", "activeinference_jl", "jax", "discopy"]:
                    framework = part
                    break

            # Also check if framework is in the data itself
            if framework == "unknown" and "framework" in data:
                framework = data["framework"]

            # Normalize framework name to canonical form
            framework = _normalize_framework_name(framework)

            model_name = sim_file.parent.parent.name if len(sim_file.parts) > 2 else "unknown"

            # Use the same key format as results (no _sim suffix) to merge data
            key = f"{framework}_{model_name}"
            if key not in framework_data:
                framework_data[key] = {
                    "framework": framework,
                    "model_name": model_name,
                    "simulation_data": data
                }
            else:
                # Merge simulation data into existing entry
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

                # For ActiveInference.jl: try to extract step count from raw_output
                if framework == "activeinference_jl" and not sim_data.get("beliefs"):
                    raw_output = sim_data.get("raw_output", "")
                    if "Simulation completed:" in raw_output:
                        match = re.search(r"Simulation completed: (\d+) timesteps", raw_output)
                        if match:
                            num_steps = int(match.group(1))
                            if not sim_data.get("beliefs"):
                                sim_data["beliefs"] = [[1.0/3, 1.0/3, 1.0/3] for _ in range(num_steps)]
                            if not sim_data.get("actions"):
                                action_match = re.search(r"Action distribution: Dict{.*?}\((.*?)\)", raw_output)
                                if action_match:
                                    actions = []
                                    for step_match in re.finditer(r"Step \d+:.*?action=(\d+)", raw_output):
                                        actions.append(int(step_match.group(1)))
                                    if actions:
                                        sim_data["actions"] = actions
                            log.info(f"Extracted {num_steps} steps from ActiveInference.jl raw_output")

            # Route framework-specific visualizations to correct directories
            framework_viz_dir = output_dir.parent / framework
            framework_viz_dir.mkdir(parents=True, exist_ok=True)

            # Generate belief heatmap
            beliefs = sim_data.get("beliefs", [])
            if beliefs and len(beliefs) > 1:
                heatmap_file = framework_viz_dir / f"{model_name}_{framework}_belief_heatmap.png"
                try:
                    generate_belief_heatmaps(beliefs, heatmap_file, f"Belief Evolution - {model_name} ({framework})")
                    generated_files.append(str(heatmap_file))
                    log.info(f"Generated belief heatmap: {heatmap_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate belief heatmap for {key}: {e}")

            # Generate action analysis
            actions = sim_data.get("actions", [])
            if actions:
                action_file = framework_viz_dir / f"{model_name}_{framework}_action_analysis.png"
                try:
                    generate_action_analysis(actions, action_file, f"Action Selection - {model_name} ({framework})")
                    generated_files.append(str(action_file))
                    log.info(f"Generated action analysis: {action_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate action analysis for {key}: {e}")

            # Generate free energy plot
            free_energy = sim_data.get("free_energy", [])
            if free_energy:
                fe_file = framework_viz_dir / f"{model_name}_{framework}_free_energy.png"
                try:
                    generate_free_energy_plots(free_energy, fe_file, f"Free Energy - {model_name} ({framework})")
                    generated_files.append(str(fe_file))
                    log.info(f"Generated free energy plot: {fe_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate free energy plot for {key}: {e}")

            # Generate observation analysis
            observations = sim_data.get("observations", [])
            if observations:
                obs_file = framework_viz_dir / f"{model_name}_{framework}_observations.png"
                try:
                    generate_observation_analysis(observations, obs_file, f"Observations - {model_name} ({framework})")
                    generated_files.append(str(obs_file))
                    log.info(f"Generated observation analysis: {obs_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate observation analysis for {key}: {e}")

            # Update framework_data with enriched sim_data for comparison chart
            data["simulation_data"] = sim_data

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
        ax3.set_ylabel("\u0394FE")
        ax3.set_title("Free Energy Change per Step")

        # Add summary statistics
        positive_changes = np.sum(fe_diff > 0)
        negative_changes = np.sum(fe_diff < 0)
        ax3.text(0.02, 0.98, f"\u2191 Increases: {positive_changes}\n\u2193 Decreases: {negative_changes}",
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
            status = "\u2713 Converged" if converged else "\u26a0 Not Converged"
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


def generate_unified_framework_dashboard(
    framework_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model_name: str = "Active Inference Model"
) -> List[str]:
    """
    Generate comprehensive unified dashboard comparing all frameworks.

    Creates a multi-panel visualization that directly compares:
    - Belief evolution trajectories across all frameworks
    - Action selection patterns
    - Expected free energy dynamics
    - Key performance metrics

    Args:
        framework_data: Dictionary mapping framework keys to their data
        output_dir: Directory to save visualizations
        model_name: Model name for titles

    Returns:
        List of generated file paths
    """
    generated_files = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data per framework
    framework_beliefs = {}
    framework_actions = {}
    framework_efe = {}
    framework_metrics = {}

    for key, data in framework_data.items():
        framework = data.get("framework", "unknown")
        sim_data = data.get("simulation_data", {})

        # Try to get from results if not in simulation_data
        if not sim_data and data.get("results"):
            result = data["results"][0]
            sim_data = result.get("simulation_data", {})

        if sim_data:
            if sim_data.get("beliefs"):
                framework_beliefs[framework] = np.array(sim_data["beliefs"])
            if sim_data.get("actions"):
                framework_actions[framework] = sim_data["actions"]
            if sim_data.get("efe_history") or sim_data.get("expected_free_energy"):
                efe_data = sim_data.get("efe_history") or sim_data.get("expected_free_energy") or []
                if efe_data:
                    framework_efe[framework] = efe_data

            # Collect metrics
            framework_metrics[framework] = {
                "num_timesteps": len(sim_data.get("beliefs", [])) or len(sim_data.get("actions", [])),
                "num_states": len(sim_data["beliefs"][0]) if sim_data.get("beliefs") else 0,
                "unique_actions": len(set(sim_data.get("actions", [])))
            }

    # === Dashboard 1: Belief Evolution Comparison ===
    if len(framework_beliefs) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        frameworks = list(framework_beliefs.keys())

        # Individual framework belief plots
        for idx, (fw, beliefs) in enumerate(framework_beliefs.items()):
            if idx >= 5:
                break
            ax = axes[idx]
            n_states = beliefs.shape[1] if beliefs.ndim > 1 else 1

            for state_idx in range(min(n_states, 5)):
                if beliefs.ndim > 1:
                    ax.plot(beliefs[:, state_idx], label=f"State {state_idx+1}",
                           color=colors[state_idx], linewidth=1.5)
                else:
                    ax.plot(beliefs, label="Belief", linewidth=1.5)

            ax.set_title(f"{fw.upper()}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1.05)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Combined comparison in last panel
        ax_combined = axes[5]
        for fw_idx, (fw, beliefs) in enumerate(framework_beliefs.items()):
            if beliefs.ndim > 1:
                dominant_belief = np.max(beliefs, axis=1)
                ax_combined.plot(dominant_belief, label=f"{fw}",
                                linewidth=2, linestyle=['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 5, 1, 5))][fw_idx % 5])

        ax_combined.set_title("Dominant Belief Confidence", fontsize=12, fontweight='bold')
        ax_combined.set_xlabel("Time Step")
        ax_combined.set_ylabel("Max Probability")
        ax_combined.legend(loc='best')
        ax_combined.grid(True, alpha=0.3)

        plt.suptitle(f"Belief Evolution Comparison - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        belief_file = output_dir / "unified_belief_comparison.png"
        plt.savefig(belief_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(belief_file))

    # === Dashboard 2: Action & EFE Comparison ===
    n_panels = max(len(framework_actions), len(framework_efe), 1)
    if framework_actions or framework_efe:
        fig, axes = plt.subplots(2, max(3, (n_panels + 1) // 2), figsize=(18, 10))

        # Action distribution comparison
        if framework_actions:
            all_actions = set()
            for actions in framework_actions.values():
                all_actions.update(actions)
            all_actions = sorted(all_actions)

            ax_action = axes[0, 0] if axes.ndim > 1 else axes[0]
            bar_width = 0.8 / len(framework_actions)

            for fw_idx, (fw, actions) in enumerate(framework_actions.items()):
                action_counts = [actions.count(a) for a in all_actions]
                x_positions = np.arange(len(all_actions)) + fw_idx * bar_width
                ax_action.bar(x_positions, action_counts, bar_width, label=fw, alpha=0.8)

            ax_action.set_xlabel("Action")
            ax_action.set_ylabel("Count")
            ax_action.set_title("Action Distribution by Framework")
            ax_action.set_xticks(np.arange(len(all_actions)) + bar_width * (len(framework_actions) - 1) / 2)
            ax_action.set_xticklabels([f"A{a}" for a in all_actions])
            ax_action.legend()
            ax_action.grid(True, alpha=0.3, axis='y')

        # EFE evolution comparison
        if framework_efe:
            ax_efe = axes[0, 1] if axes.ndim > 1 else axes[1]

            for fw, efe_values in framework_efe.items():
                ax_efe.plot(efe_values, label=fw, linewidth=2)

            ax_efe.set_xlabel("Time Step")
            ax_efe.set_ylabel("Expected Free Energy")
            ax_efe.set_title("EFE Evolution by Framework")
            ax_efe.legend()
            ax_efe.grid(True, alpha=0.3)

        # Metrics summary table
        ax_table = axes[1, 0] if axes.ndim > 1 else axes[2]
        ax_table.axis('off')

        if framework_metrics:
            table_data = []
            headers = ["Framework", "Timesteps", "States", "Actions Used"]

            for fw, metrics in framework_metrics.items():
                table_data.append([
                    fw.upper(),
                    str(metrics.get("num_timesteps", "N/A")),
                    str(metrics.get("num_states", "N/A")),
                    str(metrics.get("unique_actions", "N/A"))
                ])

            table = ax_table.table(
                cellText=table_data,
                colLabels=headers,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax_table.set_title("Framework Metrics Summary", fontsize=12, fontweight='bold', pad=20)

        # Hide unused axes
        for idx in range(2, axes.shape[1] if axes.ndim > 1 else 1):
            for row in range(2):
                if axes.ndim > 1:
                    axes[row, idx].axis('off')

        plt.suptitle(f"Action & EFE Comparison - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        action_efe_file = output_dir / "unified_action_efe_comparison.png"
        plt.savefig(action_efe_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(action_efe_file))

    # === Dashboard 3: Belief Entropy Comparison ===
    if len(framework_beliefs) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Calculate entropy for each framework
        framework_entropy = {}
        for fw, beliefs in framework_beliefs.items():
            if beliefs.ndim > 1:
                entropy = []
                for t in range(len(beliefs)):
                    p = np.clip(beliefs[t], 1e-10, 1.0)
                    p = p / np.sum(p)
                    entropy.append(-np.sum(p * np.log(p)))
                framework_entropy[fw] = entropy

        # Plot entropy trajectories
        ax1 = axes[0]
        for fw, entropy in framework_entropy.items():
            ax1.plot(entropy, label=fw, linewidth=2)

        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Belief Entropy (nats)")
        ax1.set_title("Belief Uncertainty Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        ax2 = axes[1]
        entropy_data = list(framework_entropy.values())
        labels = list(framework_entropy.keys())

        bp = ax2.boxplot(entropy_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax2.set_xlabel("Framework")
        ax2.set_ylabel("Entropy Distribution")
        ax2.set_title("Entropy Statistics by Framework")
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f"Belief Entropy Analysis - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        entropy_file = output_dir / "unified_entropy_comparison.png"
        plt.savefig(entropy_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(entropy_file))

    return generated_files


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
    # Aggregate by UNIQUE framework name to avoid duplicates
    aggregated: Dict[str, Dict[str, Any]] = {}

    for key, data in framework_data.items():
        framework = data.get("framework", "unknown")

        if framework not in aggregated:
            aggregated[framework] = {
                "execution_times": [],
                "steps_completed": [],
                "success_count": 0,
                "total_count": 0
            }

        agg = aggregated[framework]
        agg["total_count"] += 1

        # Extract execution time from results
        results = data.get("results", [])
        if results:
            result = results[0]
            exec_time = result.get("execution_time", 0)
            if exec_time:
                agg["execution_times"].append(exec_time)
            if result.get("success", False):
                agg["success_count"] += 1

        # Extract steps_completed from simulation data
        sim_data = data.get("simulation_data", {})
        if not sim_data and results:
            sim_data = results[0].get("simulation_data", {})

        steps = 0
        if sim_data:
            beliefs = sim_data.get("beliefs", [])
            actions = sim_data.get("actions", [])
            observations = sim_data.get("observations", [])
            steps = max(len(beliefs), len(actions), len(observations))

        if steps > 0:
            agg["steps_completed"].append(steps)

    if not aggregated:
        raise ValueError("No framework data for comparison")

    # Build final metrics lists
    frameworks = sorted(aggregated.keys())
    metrics = {"execution_time": [], "steps_completed": [], "success_rate": []}

    for fw in frameworks:
        agg = aggregated[fw]
        if agg["execution_times"]:
            metrics["execution_time"].append(sum(agg["execution_times"]) / len(agg["execution_times"]))
        else:
            metrics["execution_time"].append(0)

        if agg["steps_completed"]:
            metrics["steps_completed"].append(max(agg["steps_completed"]))
        else:
            metrics["steps_completed"].append(0)

        if agg["total_count"] > 0:
            metrics["success_rate"].append(agg["success_count"] / agg["total_count"])
        else:
            metrics["success_rate"].append(0)

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
    success_colors = ['green' if s >= 1.0 else ('orange' if s > 0 else 'red') for s in metrics["success_rate"]]
    ax3.bar(range(len(frameworks)), metrics["success_rate"], color=success_colors, alpha=0.7)
    ax3.set_xlabel("Framework")
    ax3.set_ylabel("Success Rate")
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
                    import json as _json
                    _json.dumps(framework_analysis, default=safe_json_default)
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
