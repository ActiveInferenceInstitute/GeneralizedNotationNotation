#!/usr/bin/env python3
"""
PyMDP Simulation Script for Classic Active Inference POMDP Agent v1

This script was automatically generated from a GNN specification.
It uses the GNN pipeline's PyMDP execution module to run an Active Inference simulation.

Model: Classic Active Inference POMDP Agent v1
Description: 
Generated: 2026-01-05 16:36:56

State Space:
- Hidden States: 3
- Observations: 3 
- Actions: 1
"""

import sys
from pathlib import Path
import logging
import subprocess

# Ensure PyMDP is installed before importing
try:
    import pymdp
    print("✅ PyMDP is available")
except ImportError:
    print("📦 PyMDP not found - installing...")
    try:
        # Try UV first (as per project rules)
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "pymdp"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            # Fallback to pip if UV fails
            print("⚠️  UV install failed, trying pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pymdp"],
                capture_output=True,
                text=True,
                timeout=120
            )
        if result.returncode == 0:
            print("✅ PyMDP installed successfully")
            import pymdp
        else:
            print(f"❌ Failed to install PyMDP: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ PyMDP installation timed out")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error installing PyMDP: {e}")
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
    
    # GNN Specification (embedded)
    gnn_spec = {
    "name": "Classic Active Inference POMDP Agent v1",
    "model_name": "Classic Active Inference POMDP Agent v1",
    "description": "This model describes a classic Active Inference agent for a discrete POMDP:\n- One observation modality (\"state_observation\") with 3 possible outcomes.\n- One hidden state factor (\"location\") with 3 possible states.\n- The hidden state is fully controllable via 3 discrete actions.\n- The agent's preferences are encoded as log-probabilities over observations.\n- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.",
    "model_parameters": {
        "num_hidden_states": 3,
        "num_obs": 3,
        "num_actions": 1
    },
    "initialparameterization": {
        "A": [
            [
                0.9,
                0.05,
                0.05
            ],
            [
                0.05,
                0.9,
                0.05
            ],
            [
                0.05,
                0.05,
                0.9
            ]
        ],
        "B": [
            [
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
            [
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    1.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    1.0,
                    0.0,
                    0.0
                ]
            ]
        ],
        "C": [
            0.1,
            0.1,
            1.0
        ],
        "D": [
            0.33333,
            0.33333,
            0.33333
        ],
        "E": [
            0.33333,
            0.33333,
            0.33333
        ]
    },
    "variables": [
        {
            "name": "A",
            "dimensions": [
                3,
                3
            ],
            "type": "float",
            "comment": "Likelihood mapping hidden states to observations"
        },
        {
            "name": "B",
            "dimensions": [
                3,
                3,
                3
            ],
            "type": "float",
            "comment": "State transitions given previous state and action"
        },
        {
            "name": "D",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Prior over initial hidden states"
        },
        {
            "name": "s",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Current hidden state distribution"
        },
        {
            "name": "s_prime",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Next hidden state distribution"
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
            "name": "C",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Log-preferences over observations"
        },
        {
            "name": "F",
            "dimensions": [
                "\u03c0"
            ],
            "type": "float",
            "comment": "Variational Free Energy for belief updating from observations"
        },
        {
            "name": "o",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Current observation (integer index)"
        },
        {
            "name": "E",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Initial policy prior (habit) over actions"
        },
        {
            "name": "\u03c0",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Policy (distribution over actions), no planning"
        },
        {
            "name": "u",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Action taken"
        },
        {
            "name": "G",
            "dimensions": [
                "\u03c0"
            ],
            "type": "float",
            "comment": "Expected Free Energy (per policy)"
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
            "source": "s",
            "relation": "-",
            "target": "B"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "E",
            "relation": ">",
            "target": "\u03c0"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "\u03c0"
        },
        {
            "source": "\u03c0",
            "relation": ">",
            "target": "u"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "u"
        },
        {
            "source": "u",
            "relation": ">",
            "target": "s_prime"
        }
    ],
    "ontology_mapping": {
        "A": "LikelihoodMatrix",
        "B": "TransitionMatrix",
        "C": "LogPreferenceVector",
        "D": "PriorOverHiddenStates",
        "E": "Habit",
        "F": "VariationalFreeEnergy",
        "G": "ExpectedFreeEnergy",
        "s": "HiddenState",
        "s_prime": "NextHiddenState",
        "o": "Observation",
        "\u03c0": "PolicyVector # Distribution over actions",
        "u": "Action       # Chosen action",
        "t": "Time"
    }
}
    
    # Output directory
    output_dir = Path("output") / "pymdp_simulations" / "Classic Active Inference POMDP Agent v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for Classic Active Inference POMDP Agent v1")
    logger.info(f"Output directory: {output_dir}")
    
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
