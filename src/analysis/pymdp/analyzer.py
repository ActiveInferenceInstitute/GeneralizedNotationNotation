#!/usr/bin/env python3
"""
PyMDP Analyzer module for generating visualizations from execution logs.
This module decouples visualization from execution, allowing post-hoc analysis.

Architecture Note:
    This is part of the ANALYSIS step (16_analysis).
    It reads raw simulation data from the EXECUTE step (12_execute) and generates
    all visualizations here, enforcing the separation: Render → Execute → Analyze.
"""

from pathlib import Path
from typing import List
import json
import logging

# Import the visualizer from its new home in analysis.pymdp
try:
    from analysis.pymdp.visualizer import PyMDPVisualizer, save_all_visualizations
except ImportError:
    # Fallback for relative imports
    try:
        from .visualizer import PyMDPVisualizer, save_all_visualizations
    except ImportError:
        logging.getLogger(__name__).warning("Could not import PyMDPVisualizer. Visualizations will be skipped.")
        PyMDPVisualizer = None
        save_all_visualizations = None

logger = logging.getLogger("analysis.pymdp")

def generate_analysis_from_logs(execution_results_dir: Path, output_dir: Path, verbose: bool = False) -> List[str]:
    """
    Generate analysis and visualizations from execution logs.
    
    This function finds raw simulation data saved by the execute step and generates
    all visualizations for the analysis step.
    
    Args:
        execution_results_dir: Directory containing execution results (e.g., output/12_execute_output)
        output_dir: Directory to save analysis outputs (e.g., output/16_analysis_output)
        verbose: Enable verbose logging
        
    Returns:
        List of generated visualization file paths
    """
    generated_files = []

    if not execution_results_dir.exists():
        logger.warning(f"Execution results directory not found: {execution_results_dir}")
        return generated_files

    # Create output directory if needed (output_dir is already the framework-specific folder)
    viz_output_dir = output_dir
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching for PyMDP simulation results in {execution_results_dir}")

    # PRIMARY: Find simulation_results.json files (created by simple_simulation.py)
    # These contain the structured trace data for visualization
    simulation_results_files = list(execution_results_dir.glob("**/simulation_results.json"))

    if simulation_results_files:
        logger.info(f"Found {len(simulation_results_files)} simulation_results.json files")

        for results_file in simulation_results_files:
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                # Get model name from data or derive from path
                model_name = data.get('model_name', results_file.parent.name)
                framework = data.get('framework', 'unknown')

                # Skip non-PyMDP results
                if framework.lower() != 'pymdp':
                    if verbose:
                        logger.debug(f"Skipping {results_file}: framework is {framework}, not PyMDP")
                    continue

                logger.info(f"Processing PyMDP results for {model_name}")

                # Extract trace data - check both new structured format and legacy flat format
                trace = data.get('simulation_trace', {})
                beliefs = trace.get('beliefs') or data.get('beliefs', [])
                true_states = trace.get('true_states') or data.get('true_states', [])
                observations = trace.get('observations') or data.get('observations', [])
                actions = trace.get('actions') or data.get('actions', [])
                efe_history = trace.get('efe_history') or data.get('metrics', {}).get('expected_free_energy', [])

                # Get model parameters
                params = data.get('model_parameters', {})
                num_states = params.get('num_states', len(beliefs[0]) if beliefs else 3)
                num_actions = params.get('num_actions', max(actions) + 1 if actions else 1)
                num_observations = params.get('num_observations', max(observations) + 1 if observations else 3)

                # Create model-specific output directory
                model_viz_dir = viz_output_dir / model_name.replace(' ', '_')
                model_viz_dir.mkdir(parents=True, exist_ok=True)

                # Generate visualizations using save_all_visualizations
                if save_all_visualizations and beliefs:
                    viz_results = {
                        "states": true_states,
                        "beliefs": beliefs,
                        "actions": actions,
                        "observations": observations,
                        "metrics": {
                            "expected_free_energy": efe_history,
                            "belief_confidence": [max(b) for b in beliefs] if beliefs else [],
                        },
                        "num_states": num_states
                    }

                    viz_files_map = save_all_visualizations(
                        simulation_results=viz_results,
                        output_dir=model_viz_dir,
                        config={"save_dir": model_viz_dir}
                    )

                    for name, filepath in viz_files_map.items():
                        generated_files.append(str(filepath))
                        logger.info(f"  Generated: {name} -> {filepath}")

                    logger.info(f"✅ Generated {len(viz_files_map)} visualizations for {model_name}")

                # --- Additional PyMDP-specific visualizations ---
                # These use data available in simulation_results.json but not
                # covered by the generic PyMDPVisualizer.

                # Cumulative Preference Plot
                cumulative_pref = data.get('metrics', {}).get('cumulative_preference', [])
                if cumulative_pref:
                    try:
                        from ..viz_base import plt, np, MATPLOTLIB_AVAILABLE
                        if MATPLOTLIB_AVAILABLE and plt is not None:
                            fig, ax = plt.subplots(figsize=(12, 5))
                            cum_sum = np.cumsum(cumulative_pref)
                            x = range(len(cumulative_pref))
                            ax.step(x, cumulative_pref, where='mid', linewidth=2,
                                    color='#2ECC71', label='Per-Step Preference', alpha=0.7)
                            ax.fill_between(x, cumulative_pref, step='mid', alpha=0.2, color='#2ECC71')
                            ax2 = ax.twinx()
                            ax2.plot(x, cum_sum, 'o-', color='#E74C3C', linewidth=2.5,
                                     markersize=5, label='Cumulative')
                            ax2.set_ylabel("Cumulative Preference", fontweight='bold', color='#E74C3C')
                            ax.set_xlabel("Time Step", fontweight='bold')
                            ax.set_ylabel("Per-Step Preference", fontweight='bold', color='#2ECC71')
                            ax.set_title(f"PyMDP Preference Accumulation — {model_name}",
                                         fontweight='bold', fontsize=13)
                            ax.grid(True, alpha=0.3)
                            # Merge legends
                            lines1, labels1 = ax.get_legend_handles_labels()
                            lines2, labels2 = ax2.get_legend_handles_labels()
                            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
                            pref_file = model_viz_dir / "cumulative_preference.png"
                            plt.savefig(str(pref_file), dpi=300, bbox_inches='tight')
                            plt.close()
                            generated_files.append(str(pref_file))
                            logger.info(f"  Generated: cumulative_preference -> {pref_file}")
                    except Exception as e:
                        logger.warning(f"Failed to generate cumulative preference plot: {e}")

                # Observation vs True State Scatter
                if observations and true_states and len(observations) == len(true_states):
                    try:
                        from ..viz_base import plt, np, MATPLOTLIB_AVAILABLE
                        if MATPLOTLIB_AVAILABLE and plt is not None:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
                            x = range(len(observations))
                            # Left: timeline comparison
                            ax1.scatter(x, true_states, label="True States", s=80,
                                        marker='D', color='#3498DB', zorder=5, alpha=0.8)
                            ax1.scatter(x, observations, label="Observations", s=50,
                                        marker='o', color='#E74C3C', zorder=4, alpha=0.7)
                            ax1.set_xlabel("Time Step", fontweight='bold')
                            ax1.set_ylabel("State / Observation Index", fontweight='bold')
                            ax1.set_title("Observations vs True States", fontweight='bold', fontsize=13)
                            ax1.legend(fontsize=10)
                            ax1.grid(True, alpha=0.3)

                            # Right: confusion matrix
                            n_vals = max(max(observations) + 1, max(true_states) + 1, num_observations)
                            confusion = np.zeros((n_vals, n_vals))
                            for obs, ts in zip(observations, true_states):
                                confusion[obs][ts] += 1
                            im = ax2.imshow(confusion, cmap='Blues', origin='lower')
                            ax2.set_xlabel("True State", fontweight='bold')
                            ax2.set_ylabel("Observation", fontweight='bold')
                            ax2.set_title("Observation Confusion Matrix", fontweight='bold', fontsize=13)
                            plt.colorbar(im, ax=ax2, label='Count')
                            for i in range(n_vals):
                                for j in range(n_vals):
                                    val = int(confusion[i][j])
                                    if val > 0:
                                        ax2.text(j, i, str(val), ha='center', va='center',
                                                 fontweight='bold', fontsize=12,
                                                 color='white' if val > confusion.max()/2 else 'black')

                            plt.suptitle(f"PyMDP Observation Analysis — {model_name}",
                                         fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            obs_file = model_viz_dir / "obs_vs_true_state.png"
                            plt.savefig(str(obs_file), dpi=300, bbox_inches='tight')
                            plt.close()
                            generated_files.append(str(obs_file))
                            logger.info(f"  Generated: obs_vs_true_state -> {obs_file}")
                    except Exception as e:
                        logger.warning(f"Failed to generate obs vs true state plot: {e}")

                else:
                    # Fallback: use PyMDPVisualizer directly for individual plots
                    if PyMDPVisualizer:
                        viz = PyMDPVisualizer(output_dir=model_viz_dir, show_plots=False)

                        # Generate discrete states visualization
                        if true_states:
                            try:
                                save_path = model_viz_dir / "discrete_states.png"
                                viz.plot_discrete_states(true_states, num_states, save_path=save_path)
                                generated_files.append(str(save_path))
                                logger.info(f"  Generated: discrete_states -> {save_path}")
                            except Exception as e:
                                logger.warning(f"Failed to generate discrete states plot: {e}")

                        # Generate belief evolution visualization
                        if beliefs:
                            try:
                                beliefs_np = [np.array(b) for b in beliefs]
                                save_path = model_viz_dir / "belief_evolution.png"
                                viz.plot_belief_evolution(beliefs_np, save_path=save_path)
                                generated_files.append(str(save_path))
                                logger.info(f"  Generated: belief_evolution -> {save_path}")
                            except Exception as e:
                                logger.warning(f"Failed to generate belief evolution plot: {e}")

                        # Generate performance metrics visualization
                        if efe_history:
                            try:
                                metrics = {
                                    "expected_free_energy": efe_history,
                                    "belief_confidence": [max(b) for b in beliefs] if beliefs else []
                                }
                                save_path = model_viz_dir / "performance_metrics.png"
                                viz.plot_performance_metrics(metrics, save_path=save_path)
                                generated_files.append(str(save_path))
                                logger.info(f"  Generated: performance_metrics -> {save_path}")
                            except Exception as e:
                                logger.warning(f"Failed to generate performance metrics plot: {e}")

                        # Generate action sequence visualization
                        if actions:
                            try:
                                save_path = model_viz_dir / "action_sequence.png"
                                viz.plot_action_sequence(actions, num_actions=num_actions, save_path=save_path)
                                generated_files.append(str(save_path))
                                logger.info(f"  Generated: action_sequence -> {save_path}")
                            except Exception as e:
                                logger.warning(f"Failed to generate action sequence plot: {e}")

                        viz.close_all_plots()

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {results_file}: {e}")
            except Exception as e:
                logger.error(f"Failed to process {results_file}: {e}")
                if verbose:
                    import traceback
                    logger.debug(traceback.format_exc())
    else:
        # FALLBACK: Search for legacy trace files
        logger.info("No simulation_results.json found, searching for legacy trace files...")
        pymdp_dirs = list(execution_results_dir.glob("*/pymdp"))

        if not pymdp_dirs:
            pymdp_dirs = list(execution_results_dir.glob("**/pymdp"))

        logger.info(f"Found {len(pymdp_dirs)} PyMDP execution directories for analysis")

        for pymdp_dir in pymdp_dirs:
            model_name = pymdp_dir.parent.name
            logger.info(f"Processing execution data for model: {model_name}")

            # Look for simulation data/trace files
            simulation_data_dir = pymdp_dir / "simulation_data"
            execution_logs_dir = pymdp_dir / "execution_logs"

            trace_files = []
            if simulation_data_dir.exists():
                trace_files.extend(list(simulation_data_dir.glob("*_trace.json")))
                trace_files.extend(list(simulation_data_dir.glob("*_simulation_data.json")))

            if execution_logs_dir.exists() and not trace_files:
                json_results = list(execution_logs_dir.glob("*_results.json"))
                trace_files.extend(json_results)

            if not trace_files:
                output_files = list(pymdp_dir.glob("*_output.txt"))
                if output_files:
                    logger.info(f"Found {len(output_files)} output file(s) for {model_name}, but no structured trace data for visualization")
                else:
                    logger.warning(f"No trace/data files found for {model_name} in {pymdp_dir}")
                continue

            for trace_file in trace_files:
                try:
                    with open(trace_file, 'r') as f:
                        data = json.load(f)

                    sim_data = data.get("simulation_data", data)

                    if not sim_data or not isinstance(sim_data, dict):
                        if verbose:
                            logger.debug(f"Skipping {trace_file}: no dictionary data found")
                        continue

                    if not any(k in sim_data for k in ['history', 'observation_history', 'belief_history', 'metrics', 'beliefs']):
                        if verbose:
                            logger.debug(f"Skipping {trace_file}: missing history keys")
                        continue

                    model_viz_dir = viz_output_dir / model_name
                    model_viz_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(f"Generating visualizations for {trace_file.name} -> {model_viz_dir}")

                    if PyMDPVisualizer:
                        viz = PyMDPVisualizer(output_dir=model_viz_dir, show_plots=False)
                        history = sim_data

                        if 'belief_history' in history or 'beliefs' in history:
                            beliefs = history.get('belief_history') or history.get('beliefs', [])
                            if beliefs:
                                beliefs_np = [np.array(b) for b in beliefs]
                                save_path = model_viz_dir / "beliefs.png"
                                viz.plot_belief_evolution(beliefs_np, title=f"{model_name} Beliefs", save_path=save_path)
                                generated_files.append(str(save_path))

                        viz.close_all_plots()

                except Exception as e:
                    logger.error(f"Failed to generate visualizations for {trace_file}: {e}")

    logger.info(f"PyMDP analysis complete: generated {len(generated_files)} visualization files")
    return generated_files

