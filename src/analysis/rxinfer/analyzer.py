"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Try to import visualization dependencies
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None


def generate_analysis_from_logs(
    execution_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> List[str]:
    """
    Generate analysis and visualizations from RxInfer execution logs.
    
    Args:
        execution_dir: Directory containing execution results
        output_dir: Directory to save visualizations
        verbose: Enable verbose logging
        
    Returns:
        List of generated visualization file paths
    """
    visualizations = []
    
    try:
        # Find RxInfer execution results
        rxinfer_dirs = list(execution_dir.glob("*/rxinfer"))
        
        for rxinfer_dir in rxinfer_dirs:
            sim_data_dir = rxinfer_dir / "simulation_data"
            if sim_data_dir.exists():
                # Load simulation results
                results_files = list(sim_data_dir.glob("*simulation_results.json"))
                for results_file in results_files:
                    try:
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        
                        model_name = rxinfer_dir.parent.name
                        viz_files = create_rxinfer_visualizations(
                            data, output_dir, model_name, verbose
                        )
                        visualizations.extend(viz_files)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {results_file}: {e}")
                        
    except Exception as e:
        logger.error(f"RxInfer analysis failed: {e}")
        
    return visualizations


def create_rxinfer_visualizations(
    data: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    verbose: bool = False
) -> List[str]:
    """
    Create visualizations from RxInfer simulation data.
    
    Args:
        data: Simulation results dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging
        
    Returns:
        List of generated file paths
    """
    visualizations = []
    
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping RxInfer visualizations")
        return visualizations
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    beliefs = data.get("beliefs", [])
    observations = data.get("observations", [])
    true_states = data.get("true_states", [])
    time_steps = data.get("time_steps", len(beliefs) if beliefs else 0)
    num_states = data.get("num_states", 3)
    
    # 1. Belief Evolution Plot
    if beliefs:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            beliefs_arr = np.array(beliefs)
            
            if beliefs_arr.ndim == 2:
                for i in range(beliefs_arr.shape[1]):
                    ax.plot(beliefs_arr[:, i], label=f"State {i+1}", linewidth=2)
            else:
                ax.plot(beliefs_arr, label="Belief", linewidth=2)
            
            ax.set_xlabel("Time Step", fontweight='bold')
            ax.set_ylabel("Belief Probability", fontweight='bold')
            ax.set_title(f"RxInfer Belief Evolution - {model_name}", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            viz_file = output_dir / f"{model_name}_rxinfer_belief_evolution.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief evolution: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief plot: {e}")
    
    # 2. Observation vs True State Plot
    if observations and true_states:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            x = range(len(observations))
            ax.scatter(x, observations, label="Observations", alpha=0.7, s=50)
            ax.scatter(x, true_states, label="True States", alpha=0.7, s=50, marker='x')
            ax.set_xlabel("Time Step", fontweight='bold')
            ax.set_ylabel("State/Observation", fontweight='bold')
            ax.set_title(f"RxInfer Observations vs True States - {model_name}", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            viz_file = output_dir / f"{model_name}_rxinfer_obs_vs_true.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated obs vs true: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create obs plot: {e}")
    
    # 3. Belief Heatmap (2D visualization of beliefs over time)
    if beliefs:
        try:
            beliefs_arr = np.array(beliefs)
            if beliefs_arr.ndim == 2 and beliefs_arr.shape[0] > 1:
                fig, ax = plt.subplots(figsize=(14, 5))
                im = ax.imshow(beliefs_arr.T, aspect='auto', cmap='viridis', 
                              origin='lower', interpolation='nearest')
                ax.set_xlabel("Time Step", fontweight='bold')
                ax.set_ylabel("State", fontweight='bold')
                ax.set_title(f"RxInfer Belief Heatmap - {model_name}", fontweight='bold')
                ax.set_yticks(range(beliefs_arr.shape[1]))
                ax.set_yticklabels([f"State {i+1}" for i in range(beliefs_arr.shape[1])])
                
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Belief Probability", fontweight='bold')
                
                viz_file = output_dir / f"{model_name}_rxinfer_belief_heatmap.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(viz_file))
                logger.info(f"Generated belief heatmap: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief heatmap: {e}")
    
    # 4. Belief Entropy (uncertainty tracking over time)
    if beliefs:
        try:
            beliefs_arr = np.array(beliefs)
            if beliefs_arr.ndim == 2:
                # Calculate entropy for each timestep: H = -sum(p * log(p))
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                beliefs_clipped = np.clip(beliefs_arr, epsilon, 1.0)
                entropy = -np.sum(beliefs_clipped * np.log2(beliefs_clipped), axis=1)
                
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(entropy, 'purple', linewidth=2, marker='o', markersize=3)
                ax.fill_between(range(len(entropy)), entropy, alpha=0.3, color='purple')
                ax.set_xlabel("Time Step", fontweight='bold')
                ax.set_ylabel("Belief Entropy (bits)", fontweight='bold')
                ax.set_title(f"RxInfer Belief Uncertainty - {model_name}", fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add max entropy line for reference
                max_entropy = np.log2(beliefs_arr.shape[1])
                ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5, 
                          label=f'Max Entropy ({max_entropy:.2f})')
                ax.legend()
                
                viz_file = output_dir / f"{model_name}_rxinfer_belief_entropy.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(viz_file))
                logger.info(f"Generated belief entropy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create entropy plot: {e}")
    
    # 5. Inference Accuracy (if we have true states)
    if beliefs and true_states:
        try:
            beliefs_arr = np.array(beliefs)
            if beliefs_arr.ndim == 2:
                # Get most likely state from beliefs (argmax)
                inferred_states = np.argmax(beliefs_arr, axis=1) + 1  # 1-indexed like true_states
                true_arr = np.array(true_states[:len(inferred_states)])
                
                # Calculate accuracy
                matches = (inferred_states == true_arr).astype(int)
                cumulative_accuracy = np.cumsum(matches) / (np.arange(len(matches)) + 1)
                
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(cumulative_accuracy * 100, 'green', linewidth=2)
                ax.fill_between(range(len(cumulative_accuracy)), cumulative_accuracy * 100, 
                               alpha=0.3, color='green')
                ax.set_xlabel("Time Step", fontweight='bold')
                ax.set_ylabel("Cumulative Accuracy (%)", fontweight='bold')
                ax.set_title(f"RxInfer Inference Accuracy - {model_name}", fontweight='bold')
                ax.set_ylim(0, 105)
                ax.grid(True, alpha=0.3)
                
                # Add final accuracy annotation
                final_acc = cumulative_accuracy[-1] * 100 if len(cumulative_accuracy) > 0 else 0
                ax.axhline(y=final_acc, color='navy', linestyle='--', alpha=0.5)
                ax.text(len(cumulative_accuracy) - 1, final_acc + 3, 
                       f'Final: {final_acc:.1f}%', ha='right', fontweight='bold')
                
                viz_file = output_dir / f"{model_name}_rxinfer_accuracy.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(viz_file))
                logger.info(f"Generated inference accuracy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create accuracy plot: {e}")
    
    return visualizations




def extract_simulation_data(execution_dir: Path, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Extract RxInfer simulation data from execution outputs.
    
    Args:
        execution_dir: Directory containing execution results
        logger: Logger instance
        
    Returns:
        Dictionary with extracted simulation data
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    data = {
        "beliefs": [],
        "observations": [],
        "true_states": [],
        "time_steps": 0,
        "model_name": "",
        "framework": "rxinfer"
    }
    
    try:
        sim_data_dir = execution_dir / "simulation_data"
        if sim_data_dir.exists():
            results_files = list(sim_data_dir.glob("*simulation_results.json"))
            if results_files:
                with open(results_files[0], 'r') as f:
                    results = json.load(f)
                data.update(results)
                
    except Exception as e:
        logger.warning(f"Failed to extract RxInfer data: {e}")
    
    return data


__all__ = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "extract_simulation_data",
]
