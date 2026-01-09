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
from typing import Dict, Any, List, Optional
import json
import logging
import numpy as np

# Import the visualizer from its new home in analysis
try:
    from analysis.pymdp_visualizer import PyMDPVisualizer, save_all_visualizations
except ImportError:
    # Fallback for relative imports
    try:
        from .pymdp_visualizer import PyMDPVisualizer, save_all_visualizations
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

    # Create visualizations directory in output_dir
    viz_output_dir = output_dir / "pymdp_visualizations"
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
        pymdp_dirs = list(execution_results_dir.glob("*/pymdp")) + list(execution_results_dir.glob("*/pymdp_gen"))
        
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

