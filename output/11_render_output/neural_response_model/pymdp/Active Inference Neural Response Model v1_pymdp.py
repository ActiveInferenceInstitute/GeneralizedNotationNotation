#!/usr/bin/env python3
"""
PyMDP Simulation Script for Active Inference Neural Response Model v1

This script was automatically generated from a GNN specification.
It uses the GNN pipeline's PyMDP execution module to run an Active Inference simulation.

Model: Active Inference Neural Response Model v1
Description: 
Generated: 2025-10-02 10:52:23

State Space:
- Hidden States: 3
- Observations: 3 
- Actions: 3
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.execute.pymdp import execute_pymdp_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main simulation function."""
    
    # GNN Specification (embedded)
    gnn_spec = {
    "name": "Active Inference Neural Response Model v1",
    "model_name": "Active Inference Neural Response Model v1",
    "description": "This model describes how a neuron responds to stimuli using Active Inference principles:\n- One primary observation modality (firing_rate) with 4 possible activity levels\n- Two auxiliary observation modalities (postsynaptic_potential, calcium_signal) for comprehensive monitoring\n- Five hidden state factors representing different aspects of neural computation\n- Three control factors for plasticity, channel modulation, and metabolic allocation\n- The model captures key neural phenomena: membrane potential dynamics, synaptic plasticity (STDP-like), activity-dependent adaptation, homeostatic regulation, and metabolic constraints\n- Preferences encode biologically realistic goals: stable firing rates, energy efficiency, and synaptic balance",
    "model_parameters": {
        "num_hidden_states": 3,
        "num_obs": 3,
        "num_actions": 3
    },
    "initialparameterization": {
        "A": [
            [
                0.05,
                0.15,
                0.25,
                0.55,
                0.4,
                0.4,
                0.2,
                0.1,
                0.35,
                0.55,
                0.3,
                0.45
            ],
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.35,
                0.45,
                0.2,
                0.15,
                0.4,
                0.45,
                0.25,
                0.4
            ],
            [
                0.15,
                0.25,
                0.35,
                0.25,
                0.3,
                0.5,
                0.2,
                0.2,
                0.45,
                0.35,
                0.2,
                0.35
            ]
        ],
        "B": [
            ""
        ],
        "C": [
            0.1,
            0.2,
            0.4,
            0.3,
            0.15,
            0.35,
            0.5,
            0.25,
            0.35,
            0.4,
            0.25,
            0.2
        ],
        "D": [
            [
                0.05,
                0.15,
                0.35,
                0.35,
                0.1,
                "# V_m distribution 0.20",
                0.4,
                0.3,
                0.1,
                "# W distribution 0.40",
                0.4,
                0.2,
                "# A distribution 0.20",
                0.6,
                0.2,
                "# H distribution 0.15",
                0.7,
                0.15
            ]
        ],
        "E": [
            0.2,
            0.3,
            0.5,
            0.25,
            0.5,
            0.25,
            0.25,
            0.5,
            0.25,
            0.3,
            0.4,
            0.3,
            0.25,
            0.5,
            0.25,
            0.3,
            0.4,
            0.3,
            0.35,
            0.4,
            0.25,
            0.3,
            0.45,
            0.25,
            0.35,
            0.4,
            0.25
        ]
    },
    "variables": [
        {
            "name": "A",
            "dimensions": [
                12,
                405
            ],
            "type": "float",
            "comment": "12 observations x 405 hidden state combinations (likelihood mapping)"
        },
        {
            "name": "B",
            "dimensions": [
                405,
                405,
                27
            ],
            "type": "float",
            "comment": "State transitions given previous state and action (5 state factors, 3 control factors)"
        },
        {
            "name": "D",
            "dimensions": [
                405
            ],
            "type": "float",
            "comment": "Prior over initial hidden states"
        },
        {
            "name": "V_m",
            "dimensions": [
                5,
                1
            ],
            "type": "float",
            "comment": "Membrane potential state (5 levels: hyperpolarized, resting, depolarized, threshold, refractory)"
        },
        {
            "name": "W",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Synaptic weight factor (4 levels: weak, moderate, strong, saturated)"
        },
        {
            "name": "A",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Adaptation state (3 levels: low, medium, high adaptation)"
        },
        {
            "name": "H",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Homeostatic set point (3 levels: low, target, high firing rate)"
        },
        {
            "name": "M",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Metabolic state (3 levels: depleted, adequate, surplus)"
        },
        {
            "name": "FR",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Firing rate (4 levels: silent, low, moderate, high)"
        },
        {
            "name": "PSP",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Postsynaptic potential (3 levels: inhibitory, none, excitatory)"
        },
        {
            "name": "Ca",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Calcium signal (3 levels: low, medium, high)"
        },
        {
            "name": "t",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Discrete time step (milliseconds scale)"
        },
        {
            "name": "C",
            "dimensions": [
                12
            ],
            "type": "float",
            "comment": "Log-preferences over observations"
        },
        {
            "name": "E",
            "dimensions": [
                27
            ],
            "type": "float",
            "comment": "Initial policy prior (habit) over actions"
        },
        {
            "name": "P",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Plasticity control (3 actions: LTD, no change, LTP)"
        },
        {
            "name": "C_mod",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Channel modulation (3 actions: decrease, maintain, increase conductance)"
        },
        {
            "name": "M_alloc",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Metabolic allocation (3 actions: conserve, balance, invest)"
        },
        {
            "name": "G",
            "dimensions": [
                "P"
            ],
            "type": "float",
            "comment": "Expected Free Energy (per policy)"
        }
    ],
    "connections": [
        {
            "source": "D",
            "relation": ">",
            "target": "V_m"
        },
        {
            "source": "V_m",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "W",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "A",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "H",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "M",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "V_m",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "W",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "V_m",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "P",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "C_mod",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "M_alloc",
            "relation": ">",
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
            "target": "P"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "P"
        },
        {
            "source": "P",
            "relation": ">",
            "target": "C_mod"
        },
        {
            "source": "C_mod",
            "relation": ">",
            "target": "M_alloc"
        }
    ],
    "ontology_mapping": {
        "A": "AdaptationState",
        "B": "TransitionMatrices",
        "C": "LogPreferenceVector",
        "D": "PriorOverHiddenStates",
        "E": "HabitVector",
        "F": "VariationalFreeEnergy",
        "G": "ExpectedFreeEnergy",
        "V_m": "MembranePotentialState",
        "W": "SynapticWeightFactor",
        "H": "HomeostaticSetPoint",
        "M": "MetabolicState",
        "FR": "FiringRateObservation",
        "PSP": "PostsynapticPotentialObservation",
        "Ca": "CalciumSignalObservation",
        "P": "PlasticityControl",
        "C_mod": "ChannelModulation",
        "M_alloc": "MetabolicAllocation",
        "t": "TimeStep"
    }
}
    
    # Configuration overrides (can be modified)
    config_overrides = {
        'num_episodes': 10,
        'max_steps_per_episode': 20,
        'planning_horizon': 5,
        'verbose_output': True,
        'save_visualizations': True,
        'random_seed': 42
    }
    
    # Output directory
    output_dir = Path("output") / "pymdp_simulations" / "Active Inference Neural Response Model v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for Active Inference Neural Response Model v1")
    logger.info(f"Output directory: {output_dir}")
    
    # Run simulation
    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            config_overrides=config_overrides
        )
        
        if success:
            logger.info("✓ Simulation completed successfully!")
            logger.info(f"Results summary:")
            logger.info(f"  Episodes: {results.get('total_episodes', 'N/A')}")
            logger.info(f"  Success Rate: {results.get('success_rate', 0):.2%}")
            logger.info(f"  Output: {results.get('output_directory', output_dir)}")
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
