#!/usr/bin/env python3
"""
PyMDP Simulation Script for Simple Markov Chain

This script was automatically generated from a GNN specification.
It uses the GNN pipeline's PyMDP execution module to run an Active Inference simulation.

Model: Simple Markov Chain
Description: 
Generated: 2026-03-24 13:58:20

State Space:
- Hidden States: 3
- Observations: 3 
- Actions: 1

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

# Remove script directory from sys.path if named 'pymdp' — it would mask the installed library
if sys.path[0] and sys.path[0].endswith("pymdp"):
    print(f"⚠️  Detected namespace conflict with script directory '{sys.path[0]}', removing from sys.path")
    sys.path.pop(0)
import logging
import subprocess
import json
import numpy as np

# Note: package is 'inferactively-pymdp', not 'pymdp'
try:
    import pymdp
    try:
        from pymdp.agent import Agent
        print("✅ PyMDP (inferactively-pymdp) is available")
    except ImportError:
        if hasattr(pymdp, "Agent"):
            print("✅ PyMDP (flat structure) is available")
        else:
            print("⚠️  PyMDP found but wrong version — install: uv pip install inferactively-pymdp")
except ImportError:
    print("❌ PyMDP not found — install: uv pip install inferactively-pymdp")
    sys.exit(1)

# Resolve repository root: prefer GNN_PROJECT_ROOT (set by Step 12), else walk up for pyproject.toml + src/
_gnn_root = os.environ.get("GNN_PROJECT_ROOT")
if _gnn_root:
    _repo = Path(_gnn_root).resolve()
    sys.path.insert(0, str(_repo))
else:
    _cur = Path(__file__).resolve().parent
    _found = None
    for _ in range(24):
        if (_cur / "pyproject.toml").is_file() and (_cur / "src").is_dir():
            _found = _cur
            break
        if _cur.parent == _cur:
            break
        _cur = _cur.parent
    if _found is None:
        print(
            "❌ Cannot locate GNN repository root. Run via the pipeline execute step, or set "
            "GNN_PROJECT_ROOT to the checkout root (directory containing pyproject.toml and src/).",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.path.insert(0, str(_found))

from src.execute.pymdp import execute_pymdp_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main simulation function."""
    
    # State Space Matrices (extracted from GNN and embedded here)
    A_matrix_data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # Likelihood matrix P(o|s)
    B_matrix_data = [[0.7, 0.3, 0.1], [0.2, 0.4, 0.3], [0.1, 0.3, 0.6]]  # Transition matrix P(s'|s,u)
    C_vector_data = [0.0, 0.0, 0.0]  # Preferences over observations
    D_vector_data = [0.5, 0.3, 0.2]  # Prior beliefs over states
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
        # B matrix normalization is handled by simple_simulation.py which knows the dimension layout.
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
    "name": "Simple Markov Chain",
    "model_name": "Simple Markov Chain",
    "description": "This model describes a minimal discrete-time Markov Chain:\n- 3 states representing weather (sunny, cloudy, rainy).\n- No actions \u2014 the system evolves passively.\n- Observations = states directly (identity mapping for monitoring).\n- Stationary transition matrix with realistic weather dynamics.\n- Tests the simplest model structure: passive state evolution with no control.",
    "model_parameters": {
        "num_hidden_states": 3,
        "num_obs": 3,
        "num_actions": 1,
        "simulation_params": {},
        "num_timesteps": 15
    },
    "initialparameterization": {
        "A": [
            [
                1.0,
                0.0,
                0.0
            ],
            [
                0.0,
                1.0,
                0.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ],
        "B": [
            [
                0.7,
                0.3,
                0.1
            ],
            [
                0.2,
                0.4,
                0.3
            ],
            [
                0.1,
                0.3,
                0.6
            ]
        ],
        "C": [
            0.0,
            0.0,
            0.0
        ],
        "D": [
            0.5,
            0.3,
            0.2
        ]
    },
    "variables": [
        {
            "name": "D",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Prior over initial states"
        },
        {
            "name": "s",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Current state distribution"
        },
        {
            "name": "s_prime",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Next state distribution"
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
            "name": "A",
            "dimensions": [
                3,
                3
            ],
            "type": "float",
            "comment": "Observation model (identity for direct monitoring)"
        },
        {
            "name": "o",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Current observation"
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
            "source": "A",
            "relation": "-",
            "target": "o"
        },
        {
            "source": "s",
            "relation": ">",
            "target": "s_prime"
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
        }
    ],
    "ontology_mapping": {
        "A": "EmissionMatrix",
        "B": "TransitionMatrix",
        "D": "InitialStateDistribution",
        "s": "HiddenState",
        "s_prime": "NextHiddenState",
        "o": "Observation",
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
    output_dir = Path("output") / "pymdp_simulations" / "Simple Markov Chain"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for Simple Markov Chain")
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
