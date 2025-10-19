#!/usr/bin/env python3
"""
Simple PyMDP Simulation based on official tutorial
Implements a basic active inference agent using PyMDP
"""

import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, Tuple, List

# Matplotlib for visualizations
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def generate_pymdp_visualizations(
    beliefs: List[np.ndarray],
    observations: List[int],
    actions: List[int],
    A: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    output_dir: Path,
    model_name: str = "pymdp_model"
) -> List[str]:
    """
    Generate comprehensive PyMDP-specific visualizations.
    
    Returns list of generated file paths.
    """
    generated_files = []
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    try:
        # 1. Belief Evolution Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        belief_array = np.array(beliefs)  # shape: (timesteps, num_states)
        timesteps = range(len(beliefs))
        
        for state_idx in range(belief_array.shape[1]):
            ax.plot(timesteps, belief_array[:, state_idx], 
                   marker='o', linewidth=2, markersize=8,
                   label=f'State {state_idx}', alpha=0.8)
        
        ax.set_xlabel('Timestep', fontsize=12, fontweight='bold')
        ax.set_ylabel('Belief (Posterior Probability)', fontsize=12, fontweight='bold')
        ax.set_title(f'PyMDP Belief Evolution Over Time\nModel: {model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add observation markers
        for t, obs in enumerate(observations):
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(t, -0.05, f'O{obs}', ha='center', va='top', 
                   fontsize=8, color='gray')
        
        belief_file = viz_dir / f"{model_name}_belief_evolution.png"
        plt.tight_layout()
        plt.savefig(belief_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(belief_file))
        logger.info(f"✅ Generated belief evolution plot: {belief_file}")
        
        # 2. Action Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Action sequence
        ax1.plot(range(len(actions)), actions, marker='s', linewidth=2, 
                markersize=10, color='coral', alpha=0.8)
        ax1.set_xlabel('Timestep', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Action Selected', fontsize=12, fontweight='bold')
        ax1.set_title('Action Selection Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks(range(max(actions) + 1))
        
        # Action histogram
        unique_actions, counts = np.unique(actions, return_counts=True)
        ax2.bar(unique_actions, counts, color='coral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Action', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Action Distribution', fontsize=12, fontweight='bold')
        ax2.set_xticks(unique_actions)
        ax2.grid(True, alpha=0.3, axis='y')
        
        action_file = viz_dir / f"{model_name}_action_analysis.png"
        plt.tight_layout()
        plt.savefig(action_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(action_file))
        logger.info(f"✅ Generated action analysis plot: {action_file}")
        
        # 3. Observation Model (A matrix) Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(A, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'P(obs|state)'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        ax.set_xlabel('Hidden State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Observation', fontsize=12, fontweight='bold')
        ax.set_title(f'PyMDP Observation Model (A Matrix)\nLikelihood: P(observation | state)', 
                    fontsize=12, fontweight='bold', pad=15)
        
        a_matrix_file = viz_dir / f"{model_name}_A_matrix.png"
        plt.tight_layout()
        plt.savefig(a_matrix_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(a_matrix_file))
        logger.info(f"✅ Generated A matrix heatmap: {a_matrix_file}")
        
        # 4. Preference Vector (C) and Prior (D)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Preferences
        x_pos = np.arange(len(C))
        ax1.bar(x_pos, C, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Observation', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Preference (log probability)', fontsize=12, fontweight='bold')
        ax1.set_title('Observation Preferences (C vector)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Prior
        x_pos = np.arange(len(D))
        ax2.bar(x_pos, D, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('State', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Prior Probability', fontsize=12, fontweight='bold')
        ax2.set_title('State Prior (D vector)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        prefs_file = viz_dir / f"{model_name}_preferences_prior.png"
        plt.tight_layout()
        plt.savefig(prefs_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(prefs_file))
        logger.info(f"✅ Generated preferences/prior plot: {prefs_file}")
        
        # 5. Belief State Space Trajectory
        if belief_array.shape[1] == 3:  # Only for 3-state models
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectory
            ax.plot(belief_array[:, 0], belief_array[:, 1], belief_array[:, 2],
                   'o-', linewidth=2, markersize=8, alpha=0.7, color='purple')
            
            # Mark start and end
            ax.scatter(*belief_array[0], s=200, c='green', marker='*', 
                      edgecolors='black', linewidths=2, label='Start', zorder=5)
            ax.scatter(*belief_array[-1], s=200, c='red', marker='X', 
                      edgecolors='black', linewidths=2, label='End', zorder=5)
            
            ax.set_xlabel('P(State 0)', fontsize=11, fontweight='bold')
            ax.set_ylabel('P(State 1)', fontsize=11, fontweight='bold')
            ax.set_zlabel('P(State 2)', fontsize=11, fontweight='bold')
            ax.set_title('Belief State Space Trajectory\n(Simplex in 3D)', 
                        fontsize=13, fontweight='bold', pad=20)
            ax.legend(fontsize=10)
            
            trajectory_file = viz_dir / f"{model_name}_belief_trajectory_3d.png"
            plt.tight_layout()
            plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(trajectory_file))
            logger.info(f"✅ Generated 3D belief trajectory: {trajectory_file}")
        
        # 6. Summary Dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Belief evolution (larger plot)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        for state_idx in range(belief_array.shape[1]):
            ax1.plot(timesteps, belief_array[:, state_idx], 
                    marker='o', linewidth=2, label=f'State {state_idx}', alpha=0.8)
        ax1.set_xlabel('Timestep', fontweight='bold')
        ax1.set_ylabel('Belief', fontweight='bold')
        ax1.set_title('Belief Evolution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Action sequence
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(range(len(actions)), actions, 'o-', color='coral', linewidth=2)
        ax2.set_xlabel('Time', fontweight='bold', fontsize=9)
        ax2.set_ylabel('Action', fontweight='bold', fontsize=9)
        ax2.set_title('Actions', fontweight='bold', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Observation sequence
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(range(len(observations)), observations, 's-', color='steelblue', linewidth=2)
        ax3.set_xlabel('Time', fontweight='bold', fontsize=9)
        ax3.set_ylabel('Observation', fontweight='bold', fontsize=9)
        ax3.set_title('Observations', fontweight='bold', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # A matrix heatmap
        ax4 = fig.add_subplot(gs[2, 0])
        im = ax4.imshow(A, cmap='YlOrRd', aspect='auto')
        ax4.set_xlabel('State', fontweight='bold', fontsize=9)
        ax4.set_ylabel('Obs', fontweight='bold', fontsize=9)
        ax4.set_title('A Matrix', fontweight='bold', fontsize=10)
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # C vector
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.bar(range(len(C)), C, color='skyblue', alpha=0.7)
        ax5.set_xlabel('Observation', fontweight='bold', fontsize=9)
        ax5.set_ylabel('Preference', fontweight='bold', fontsize=9)
        ax5.set_title('C Vector', fontweight='bold', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # D vector
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.bar(range(len(D)), D, color='lightgreen', alpha=0.7)
        ax6.set_xlabel('State', fontweight='bold', fontsize=9)
        ax6.set_ylabel('Prior', fontweight='bold', fontsize=9)
        ax6.set_title('D Vector', fontweight='bold', fontsize=10)
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'PyMDP Active Inference Dashboard - {model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        dashboard_file = viz_dir / f"{model_name}_dashboard.png"
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        generated_files.append(str(dashboard_file))
        logger.info(f"✅ Generated summary dashboard: {dashboard_file}")
        
    except Exception as e:
        logger.error(f"Visualization generation error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return generated_files


def run_simple_pymdp_simulation(gnn_spec: Dict[str, Any], output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a simple PyMDP active inference simulation based on the GNN specification.
    
    This follows the pattern from the official PyMDP tutorial:
    - Create A, B, C, D matrices from GNN spec
    - Initialize agent with these matrices
    - Run inference loop
    
    Args:
        gnn_spec: GNN specification with initialparameterization
        output_dir: Directory for outputs
        
    Returns:
        Tuple of (success, results_dict)
    """
    try:
        # Import PyMDP components
        from pymdp.agent import Agent
        from pymdp import utils
        
        logger.info("Starting simple PyMDP simulation")
        
        # Extract matrices from GNN spec
        init_params = gnn_spec.get('initialparameterization', {})
        
        # Get A matrix (observation model)
        A_data = init_params.get('A', [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        A = np.array(A_data, dtype=np.float64)
        
        # Get B matrix (transition model) - simplified to identity for now
        num_states = A.shape[1]
        B = np.eye(num_states, dtype=np.float64)
        B = B[:, :, np.newaxis]  # Add action dimension
        
        # Get C vector (preferences)
        C_data = init_params.get('C', [0.1, 0.1, 1.0])
        C = np.array(C_data, dtype=np.float64)
        
        # Get D vector (prior over states)
        D_data = init_params.get('D', [0.33333, 0.33333, 0.33333])
        D = np.array(D_data, dtype=np.float64)
        
        logger.info(f"Created matrices: A={A.shape}, B={B.shape}, C={C.shape}, D={D.shape}")
        
        # PyMDP expects obj_array format (numpy object arrays)
        # Use utils.obj_array to properly wrap arrays
        A_obj = utils.obj_array(1)  # 1 observation modality
        A_obj[0] = A
        
        B_obj = utils.obj_array(1)  # 1 hidden state factor
        B_obj[0] = B
        
        C_obj = utils.obj_array(1)  # 1 observation modality
        C_obj[0] = C
        
        D_obj = utils.obj_array(1)  # 1 hidden state factor
        D_obj[0] = D
        
        logger.info("Wrapped arrays in obj_array format")
        
        # Create PyMDP agent with obj_array wrapped arrays
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
        logger.info("Successfully created PyMDP agent")
        
        # Run a simple simulation loop (increased timesteps for better visualization)
        num_timesteps = 15
        observations = []
        beliefs = []
        actions = []
        beliefs_raw = []  # Store raw numpy arrays for visualization
        
        for t in range(num_timesteps):
            # Sample observation (for demo, use random)
            obs_idx = np.random.randint(0, A.shape[0])
            obs = np.array([obs_idx])
            observations.append(obs_idx)
            
            # Infer states
            qs = agent.infer_states(obs)
            beliefs.append(qs[0].tolist())
            beliefs_raw.append(qs[0].copy())  # Store for visualization
            
            # Infer policy
            q_pi, neg_efe = agent.infer_policies()
            
            # Sample action
            action = agent.sample_action()
            actions.append(int(action[0]))
            
            logger.info(f"Step {t}: obs={obs_idx}, belief={qs[0]}, action={action[0]}")
        
        # Generate comprehensive visualizations
        logger.info("Generating PyMDP-specific visualizations...")
        model_name = gnn_spec.get('model_name', gnn_spec.get('name', 'pymdp_model'))
        viz_files = generate_pymdp_visualizations(
            beliefs=beliefs_raw,
            observations=observations,
            actions=actions,
            A=A,
            C=C,
            D=D,
            output_dir=output_dir,
            model_name=model_name
        )
        logger.info(f"✅ Generated {len(viz_files)} visualization files")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "success": True,
            "framework": "PyMDP",
            "num_timesteps": num_timesteps,
            "observations": observations,
            "beliefs": beliefs,
            "actions": actions,
            "model_parameters": {
                "A_shape": list(A.shape),
                "B_shape": list(B.shape),
                "C_shape": list(C.shape),
                "D_shape": list(D.shape)
            },
            "visualizations": {
                "count": len(viz_files),
                "files": viz_files,
                "types": [
                    "belief_evolution",
                    "action_analysis",
                    "A_matrix_heatmap",
                    "preferences_prior",
                    "belief_trajectory_3d",
                    "dashboard"
                ]
            },
            "validation": {
                "all_beliefs_valid": all(0 <= b[i] <= 1 for b in beliefs for i in range(len(b))),
                "beliefs_sum_to_one": all(abs(sum(b) - 1.0) < 0.01 for b in beliefs),
                "actions_in_range": all(0 <= a < B.shape[2] for a in actions)
            }
        }
        
        results_file = output_dir / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Saved results to {results_file}")
        logger.info(f"📊 Validation: beliefs_valid={results['validation']['all_beliefs_valid']}, "
                   f"sum_to_one={results['validation']['beliefs_sum_to_one']}")
        
        return True, results
        
    except ImportError as e:
        logger.error(f"PyMDP not available: {e}")
        return False, {
            "success": False,
            "error": f"PyMDP import failed: {str(e)}",
            "suggestion": "Install with: uv pip install inferactively-pymdp"
        }
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        return False, {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

