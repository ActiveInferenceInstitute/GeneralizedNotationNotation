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

# Try to import seaborn, fall back gracefully
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

logger = logging.getLogger(__name__)


from .pymdp_visualizer import PyMDPVisualizer, save_all_visualizations

# Re-export these for backward compatibility if needed, though they aren't used here anymore 
# as we use the imported class directly.


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
        # Import PyMDP components using modern API
        # The correct package is 'inferactively-pymdp' (uv pip install inferactively-pymdp)
        try:
            from pymdp.agent import Agent
            from pymdp import utils
            logger.info("Using PyMDP (inferactively-pymdp) API")
        except ImportError:
            Agent = None
            utils = None
        
        if Agent is None:
            # Use package detector to identify the issue
            from .package_detector import (
                detect_pymdp_installation,
                get_pymdp_installation_instructions
            )
            
            detection = detect_pymdp_installation()
            instructions = get_pymdp_installation_instructions()
            
            if detection.get("wrong_package"):
                msg = (
                    f"Wrong PyMDP package installed. {instructions}\n"
                    "The installed 'pymdp' package contains MDP/MDPSolver but not the "
                    "Active Inference Agent class required for this simulation."
                )
                logger.error(msg)
                return False, {
                    "success": False,
                    "error": msg,
                    "wrong_package": True,
                    "suggestion": "Uninstall 'pymdp' and install 'inferactively-pymdp'",
                    "install_command": "uv pip install inferactively-pymdp"
                }
            else:
                msg = f"PyMDP Agent class not found. {instructions}"
                logger.error(msg)
                return False, {
                    "success": False,
                    "error": msg,
                    "suggestion": instructions,
                    "install_command": "uv pip install inferactively-pymdp"
                }
        
        logger.info("Starting simple PyMDP simulation")
        
        # Extract matrices from GNN spec
        init_params = gnn_spec.get('initialparameterization', {})
        
        # 1. Get A matrix (observation model)
        # PyMDP expects A to be an obj_array of modalities, each a (num_obs, num_states) matrix
        A_data = init_params.get('A')
        if A_data is None:
            logger.info("A matrix not found in spec, using default identity")
            A = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], dtype=np.float64)
        else:
            A = np.array(A_data, dtype=np.float64)
        
        # 2. Get B matrix (transition model)
        # PyMDP expects B to be an obj_array of factors, each a (num_states, num_states, num_actions) matrix
        B_data = init_params.get('B')
        if B_data is None:
            logger.info("B matrix not found in spec, using default identity")
            B = np.eye(A.shape[1], dtype=np.float64)[:, :, np.newaxis]
        else:
            B_raw = np.array(B_data, dtype=np.float64)
            # GNN format for B is typically (action, prev_state, next_state)
            # PyMDP expects (next_state, prev_state, action)
            # Transpose any 3D array from (action, prev, next) to (next, prev, action)
            if B_raw.ndim == 3:
                # Transpose (0, 1, 2) -> (2, 1, 0) means action,prev,next -> next,prev,action
                B = B_raw.transpose(2, 1, 0)
                logger.info(f"Transposed B from {B_raw.shape} to {B.shape}")
            else:
                B = B_raw
                logger.warning(f"B matrix has unexpected dimensions: {B_raw.ndim}D, expected 3D")
        
        # 3. Get C vector (preferences)
        C_data = init_params.get('C')
        if C_data is None:
            C = np.zeros(A.shape[0], dtype=np.float64)
        else:
            C = np.array(C_data, dtype=np.float64).flatten()
            
        # 4. Get D vector (prior over states)
        D_data = init_params.get('D')
        if D_data is None:
            D = np.ones(A.shape[1], dtype=np.float64) / A.shape[1]
        else:
            D = np.array(D_data, dtype=np.float64).flatten()
            
        # 5. Get E vector (habit/policy prior) - optional
        E_data = init_params.get('E')
        if E_data is not None:
            E = np.array(E_data, dtype=np.float64).flatten()
        else:
            E = None
        
        logger.info(f"Created matrices: A={A.shape}, B={B.shape}, C={C.shape}, D={D.shape}")
        
        # PyMDP expects obj_array format (numpy object arrays)
        A_obj = utils.obj_array(1)
        A_obj[0] = A
        
        B_obj = utils.obj_array(1)
        B_obj[0] = B
        
        C_obj = utils.obj_array(1)
        C_obj[0] = C
        
        D_obj = utils.obj_array(1)
        D_obj[0] = D
        
        # E vector is policy-level, so pass directly (pyMDP expects 1D array over policies)
        logger.info("Wrapped arrays in obj_array format (except E)")
        
        # Create PyMDP agent
        agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj, E=E)
        logger.info("Successfully created PyMDP agent")
        
        num_timesteps = 15
        observations = []
        beliefs = []
        true_states = []
        actions = []
        beliefs_raw = []  # Store raw numpy arrays for visualization
        
        # Initial true state
        current_state = np.random.choice(range(A.shape[1]), p=D)
        
        for t in range(num_timesteps):
            # Track true state
            true_states.append(int(current_state))
            
            # Sample observation from current true state
            obs_probs = A[:, current_state]
            obs_idx = np.random.choice(range(A.shape[0]), p=obs_probs)
            obs = np.array([obs_idx])
            observations.append(int(obs_idx))
            
            # Infer states
            qs = agent.infer_states(obs)
            beliefs.append(qs[0].tolist())
            beliefs_raw.append(qs[0].copy())  # Store for visualization
            
            # Infer policy
            q_pi, neg_efe = agent.infer_policies()
            
            # Sample action
            action = agent.sample_action()
            actions.append(int(action[0]))
            
            # Transition true state using B matrix
            # B is (next_state, prev_state, action)
            next_state_probs = B[:, current_state, int(action[0])]
            current_state = np.random.choice(range(B.shape[0]), p=next_state_probs)
            
            logger.info(f"Step {t}: true_s={true_states[-1]}, obs={obs_idx}, belief={np.round(qs[0], 2)}, action={action[0]}")
        
        # Generate comprehensive visualizations
        logger.info("Generating PyMDP-specific visualizations...")
        model_name = gnn_spec.get('model_name', gnn_spec.get('name', 'pymdp_model'))
        
        # Prepare results for visualization
        viz_results = {
            "states": true_states,
            "beliefs": beliefs,
            "actions": actions,
            "observations": observations,
            "metrics": {
                "expected_free_energy": [0.0] * len(actions), # Placeholder or calculate if available
                "belief_confidence": [max(b) for b in beliefs],
                "cumulative_preference": [0.0] * len(actions) # Placeholder
            },
            "num_states": A.shape[1]
        }
        
        viz_files_map = save_all_visualizations(
            simulation_results=viz_results,
            output_dir=output_dir / "visualizations",
            config={"save_dir": output_dir / "visualizations"}
        )
        viz_files = list(viz_files_map.values())
        logger.info(f"✅ Generated {len(viz_files)} visualization files")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "success": True,
            "framework": "PyMDP",
            "num_timesteps": num_timesteps,
            "observations": observations,
            "true_states": true_states,
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
                "files": [str(f) for f in viz_files],
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

