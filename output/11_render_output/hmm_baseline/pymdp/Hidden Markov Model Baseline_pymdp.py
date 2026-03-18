#!/usr/bin/env python3
"""
PyMDP Simulation Script for Hidden Markov Model Baseline

This script was automatically generated from a GNN specification.
It uses the GNN pipeline's PyMDP execution module to run an Active Inference simulation.

Model: Hidden Markov Model Baseline
Description: 
Generated: 2026-03-17 16:47:28

State Space:
- Hidden States: 4
- Observations: 6 
- Actions: 4

State Space Matrices (from GNN):
- A (Likelihood): Present
- B (Transition): Present
- C (Preferences): Present
- D (Prior): Present
- E (Habits): Missing
"""

import sys
from pathlib import Path
import os

# Prevent import conflict with local 'pymdp' folder which contains this script
# sys.path[0] is the script directory. If it's named 'pymdp', it masks the installed library.
if sys.path[0] and sys.path[0].endswith("pymdp"):
    print(f"⚠️  Detected namespace conflict with script directory '{sys.path[0]}', removing from sys.path")
    sys.path.pop(0)
import logging
import subprocess
import json
import numpy as np

# Ensure PyMDP is installed before importing
# Note: The correct package name is 'inferactively-pymdp', not 'pymdp'
try:
    import pymdp
    # Verify it is the CORRECT pymdp (inferactively-pymdp)
    try:
        from pymdp.agent import Agent
        print("✅ PyMDP (inferactively-pymdp) is available")
    except ImportError:
        # Check if it might be the flat structure (unlikely for modern, but possible)
        if hasattr(pymdp, "Agent"):
             print("✅ PyMDP (flat structure) is available")
        else:
             print("⚠️  PyMDP package found, but it appears to be the wrong version (missing Agent).")
             print("💡 Please install the correct package: uv pip install inferactively-pymdp")
             # Proceeding anyway, might fail later but better than auto-install crash
except ImportError:
    print("❌ PyMDP not found. This script requires 'inferactively-pymdp'.")
    print("💡 Install with: uv pip install inferactively-pymdp")
    # We will not attempt auto-install as it is fragile in managed environments
    sys.exit(1)

# Add project root to path for imports (script is 5 levels deep: output/11_render_output/actinf_pomdp_agent/pymdp/script.py)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.execute.pymdp import execute_pymdp_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main simulation function."""
    
    # State Space Matrices (extracted from GNN and embedded here)
    A_matrix_data = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7], [0.1, 0.1, 0.4, 0.4], [0.4, 0.4, 0.1, 0.1]]  # Likelihood matrix P(o|s)
    B_matrix_data = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.2, 0.1], [0.1, 0.1, 0.6, 0.2], [0.1, 0.1, 0.1, 0.6]]  # Transition matrix P(s'|s,u)
    C_vector_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Preferences over observations
    D_vector_data = [0.25, 0.25, 0.25, 0.25]  # Prior beliefs over states
    E_vector_data = None  # Policy priors (habits)
    
    # Convert to numpy arrays
    if A_matrix_data is not None:
        A_matrix = np.array(A_matrix_data)
        # Normalize A matrix (columns should sum to 1)
        if A_matrix.ndim == 2:
            norm = np.sum(A_matrix, axis=0)
            A_matrix = A_matrix / np.where(norm == 0, 1, norm)
        logger.info(f"A matrix shape: {A_matrix.shape}")
    else:
        A_matrix = None
        logger.warning("A matrix not provided")
    
    if B_matrix_data is not None:
        B_matrix = np.array(B_matrix_data)
        # Normalize B matrix (columns should sum to 1)
        # B shape in PyMDP usually (next_state, prev_state, action)
        # But GNN might provide (action, prev_state, next_state) or similar.
        # Here we assume GNN provides B as [action][prev][next] or similar from JSON
        # We will trust the default simple_simulation handling for dimension/transposition,
        # but here we just ensure values are normalized along the last dimension if it sums approx to > 0.
        # Actually, let's just ensure it's normalized in simple_simulation or here?
        # Better to do it in simple_simulation.py where we know the dimensions?
        # The simple_simulation.py loads this gnn_spec.
        # So we should modify simple_simulation.py instead?
        # NO, this script IS the one that passes data to simple_simulation via gnn_spec['initialparameterization'].
        # The simple_simulation.py reads A from gnn_spec['initialparameterization']['A'].
        # So if we update 'A_matrix' variable here, we must ensure it is passed back to gnn_spec correctly.
        # Lines 494 update gnn_spec using 'A_matrix.tolist()'.
        # So normalizing HERE is correct.
        
        # However, for B matrix, dimensions are tricky. 
        # Let's simple_simulation handle B normalization since it does transposition logic.
        logger.info(f"B matrix shape: {B_matrix.shape}")
    else:
        B_matrix = None
        logger.warning("B matrix not provided")
    
    if C_vector_data is not None:
        C_vector = np.array(C_vector_data)
        logger.info(f"C vector shape: {C_vector.shape}")
    else:
        C_vector = None
        logger.warning("C vector not provided")
    
    if D_vector_data is not None:
        D_vector = np.array(D_vector_data)
        # Normalize D vector
        norm = np.sum(D_vector)
        D_vector = D_vector / np.where(norm == 0, 1, norm)
        logger.info(f"D vector shape: {D_vector.shape}")
    else:
        D_vector = None
        logger.warning("D vector not provided")
    
    if E_vector_data is not None:
        E_vector = np.array(E_vector_data)
        logger.info(f"E vector shape: {E_vector.shape}")
    else:
        E_vector = None
    
    # GNN Specification (embedded with state spaces)
    gnn_spec = {
    "name": "Hidden Markov Model Baseline",
    "model_name": "Hidden Markov Model Baseline",
    "description": "A standard discrete Hidden Markov Model with:\n- 4 hidden states with Markovian dynamics\n- 6 observation symbols\n- Fixed transition and emission matrices\n- No action selection (passive inference only)\n- Suitable for sequence modeling and state estimation tasks",
    "model_parameters": {
        "num_hidden_states": 4,
        "num_obs": 6,
        "num_actions": 4,
        "simulation_params": {},
        "num_timesteps": 15
    },
    "initialparameterization": {
        "A": [
            [
                0.7,
                0.1,
                0.1,
                0.1
            ],
            [
                0.1,
                0.7,
                0.1,
                0.1
            ],
            [
                0.1,
                0.1,
                0.7,
                0.1
            ],
            [
                0.1,
                0.1,
                0.1,
                0.7
            ],
            [
                0.1,
                0.1,
                0.4,
                0.4
            ],
            [
                0.4,
                0.4,
                0.1,
                0.1
            ]
        ],
        "B": [
            [
                0.7,
                0.1,
                0.1,
                0.1
            ],
            [
                0.1,
                0.7,
                0.2,
                0.1
            ],
            [
                0.1,
                0.1,
                0.6,
                0.2
            ],
            [
                0.1,
                0.1,
                0.1,
                0.6
            ]
        ],
        "C": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "D": [
            0.25,
            0.25,
            0.25,
            0.25
        ]
    },
    "variables": [
        {
            "name": "A",
            "dimensions": [
                6,
                4
            ],
            "type": "float",
            "comment": "Emission matrix: observations x hidden states"
        },
        {
            "name": "D",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Initial state distribution (prior)"
        },
        {
            "name": "s",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Hidden state belief (posterior)"
        },
        {
            "name": "s_prime",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Next hidden state"
        },
        {
            "name": "alpha",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Forward variable (belief propagation)"
        },
        {
            "name": "beta",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Backward variable"
        },
        {
            "name": "t",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Discrete time step"
        },
        {
            "name": "o",
            "dimensions": [
                6,
                1
            ],
            "type": "float",
            "comment": "Current observation (one-hot)"
        },
        {
            "name": "B",
            "dimensions": [
                4,
                4
            ],
            "type": "float",
            "comment": "Transition matrix (no action dependence)"
        }
    ],
    "connections": [
        {
            "source": "D",
            "relation": ">",
            "target": "s"
        },
        {
            "source": "s",
            "relation": "-",
            "target": "A"
        },
        {
            "source": "s",
            "relation": ">",
            "target": "s_prime"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "o"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "s_prime"
        },
        {
            "source": "s",
            "relation": "-",
            "target": "B"
        },
        {
            "source": "s",
            "relation": "-",
            "target": "F"
        },
        {
            "source": "o",
            "relation": "-",
            "target": "F"
        },
        {
            "source": "s",
            "relation": "-",
            "target": "alpha"
        },
        {
            "source": "o",
            "relation": "-",
            "target": "alpha"
        },
        {
            "source": "alpha",
            "relation": ">",
            "target": "s_prime"
        },
        {
            "source": "s_prime",
            "relation": "-",
            "target": "beta"
        }
    ],
    "ontology_mapping": {
        "A": "EmissionMatrix",
        "B": "TransitionMatrix",
        "D": "InitialStateDistribution",
        "s": "HiddenState",
        "s_prime": "NextHiddenState",
        "o": "Observation",
        "F": "VariationalFreeEnergy",
        "alpha": "ForwardVariable",
        "beta": "BackwardVariable",
        "t": "Time"
    }
}
    
    # Ensure state space matrices are in gnn_spec for execution
    if 'initialparameterization' not in gnn_spec:
        gnn_spec['initialparameterization'] = {}
    if A_matrix is not None:
        gnn_spec['initialparameterization']['A'] = A_matrix.tolist() if hasattr(A_matrix, 'tolist') else A_matrix
    if B_matrix is not None:
        gnn_spec['initialparameterization']['B'] = B_matrix.tolist() if hasattr(B_matrix, 'tolist') else B_matrix
    if C_vector is not None:
        gnn_spec['initialparameterization']['C'] = C_vector.tolist() if hasattr(C_vector, 'tolist') else C_vector
    if D_vector is not None:
        gnn_spec['initialparameterization']['D'] = D_vector.tolist() if hasattr(D_vector, 'tolist') else D_vector
    if E_vector is not None:
        gnn_spec['initialparameterization']['E'] = E_vector.tolist() if hasattr(E_vector, 'tolist') else E_vector
    
    # Output directory
    output_dir = Path("output") / "pymdp_simulations" / "Hidden Markov Model Baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for Hidden Markov Model Baseline")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"State space matrices: A={A_matrix is not None}, B={B_matrix is not None}, C={C_vector is not None}, D={D_vector is not None}, E={E_vector is not None}")
    
    # Run simulation
    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            correlation_id="render_generated_script"
        )
        
        if success:
            logger.info("✓ Simulation completed successfully!")
            logger.info(f"Results summary:")
            logger.info(f"  Correlation ID: {results.get('correlation_id', 'N/A')}")
            logger.info(f"  Success: {results.get('success', False)}")
            logger.info(f"  Output: {output_dir}")
            return 0
        else:
            logger.error("✗ Simulation failed!")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
