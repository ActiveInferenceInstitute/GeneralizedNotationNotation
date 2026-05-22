#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Simple MDP Agent

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Simple MDP Agent
Description:  
Generated:    2026-05-22 06:18:15

State Space:
  - Hidden States: 4
  - Observations:  4
  - Actions:       4

Initial matrices present in GNN spec:
  - A (likelihood):   Present
  - B (transitions):  Present
  - C (preferences):  Present
  - D (state prior):  Present
  - E (policy prior): Missing
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Script directory name 'pymdp' would shadow the installed library — drop it
# ---------------------------------------------------------------------------
if sys.path and sys.path[0] and sys.path[0].endswith("pymdp"):
    sys.path.pop(0)

# ---------------------------------------------------------------------------
# Repository root resolution (prefer GNN_PROJECT_ROOT; else walk upwards)
# ---------------------------------------------------------------------------
_gnn_root = os.environ.get("GNN_PROJECT_ROOT")
if _gnn_root:
    _repo = Path(_gnn_root).resolve()
    sys.path.insert(0, str(_repo / "src"))
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
            "ERROR: Cannot locate GNN repository root. Run via the pipeline "
            "execute step, or set GNN_PROJECT_ROOT to the checkout root.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.path.insert(0, str(_found / "src"))

# ---------------------------------------------------------------------------
# pymdp 1.0.0 presence check (hard requirement)
# ---------------------------------------------------------------------------
try:
    import pymdp  # noqa: F401
    from pymdp.agent import Agent  # noqa: F401
    if not hasattr(Agent, "update_empirical_prior"):
        raise ImportError("unsupported pymdp (<1.0.0) detected")
    print("PyMDP 1.0.0+ detected (JAX-first Agent).")
except ImportError as e:
    print(
        "ERROR: pymdp 1.0.0 required. Install with: "
        "uv pip install 'inferactively-pymdp>=1.0.0' (original error: "
        + str(e) + ")",
        file=sys.stderr,
    )
    sys.exit(1)

from execute.pymdp import execute_pymdp_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run a pymdp 1.0.0 simulation for the GNN model embedded in this file."""
    # Matrices embedded verbatim from the GNN spec.
    A_data = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    B_data = [[[0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9]], [[0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1]], [[0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 0.1, 0.9], [0.9, 0.1, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]], [[0.0, 0.0, 0.1, 0.9], [0.0, 0.0, 0.9, 0.1], [0.1, 0.9, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]]
    C_data = [0.0, 0.0, 0.0, 3.0]
    D_data = [0.25, 0.25, 0.25, 0.25]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Simple MDP Agent",
    "model_name": "Simple MDP Agent",
    "description": "This model describes a fully observable Markov Decision Process (MDP):\n- 4 hidden states representing grid positions (corners of a 2x2 grid).\n- Observations are identical to states (A = identity matrix).\n- 4 actions: stay, move-north, move-south, move-east.\n- Preferences strongly favor state/observation 3 (goal location).\n- Tests the degenerate POMDP case where partial observability is absent.",
    "model_parameters": {
        "num_hidden_states": 4,
        "num_obs": 4,
        "num_actions": 4,
        "num_timesteps": 25,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 2,
        "num_modalities": None,
        "state_factors": [
            {
                "name": "s",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Current hidden state distribution",
                "index": 2
            },
            {
                "name": "s_prime",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Next hidden state distribution",
                "index": 3
            }
        ],
        "observation_modalities": [],
        "control_factors": [
            {
                "name": "\u03c0",
                "size": 4,
                "dimensions": [
                    4
                ],
                "type": "float",
                "comment": "Policy (distribution over actions)",
                "index": 0
            },
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Action taken",
                "index": 1
            }
        ],
        "passive_model": False,
        "simulation_params": {}
    },
    "initialparameterization": {
        "A": [
            [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        "B": [
            [
                [
                    0.9,
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.9,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                    0.9
                ]
            ],
            [
                [
                    0.1,
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.9,
                    0.1
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.9,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                    0.9
                ],
                [
                    0.9,
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.9,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.9,
                    0.1
                ],
                [
                    0.1,
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.0,
                    0.0
                ]
            ]
        ],
        "C": [
            0.0,
            0.0,
            0.0,
            3.0
        ],
        "D": [
            0.25,
            0.25,
            0.25,
            0.25
        ]
    },
    "structured_pomdp": {
        "matrices": {
            "A": [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "B": [
                [
                    [
                        0.9,
                        0.1,
                        0.0,
                        0.0
                    ],
                    [
                        0.1,
                        0.9,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.9,
                        0.1
                    ],
                    [
                        0.0,
                        0.0,
                        0.1,
                        0.9
                    ]
                ],
                [
                    [
                        0.1,
                        0.9,
                        0.0,
                        0.0
                    ],
                    [
                        0.9,
                        0.1,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.1,
                        0.9
                    ],
                    [
                        0.0,
                        0.0,
                        0.9,
                        0.1
                    ]
                ],
                [
                    [
                        0.0,
                        0.0,
                        0.9,
                        0.1
                    ],
                    [
                        0.0,
                        0.0,
                        0.1,
                        0.9
                    ],
                    [
                        0.9,
                        0.1,
                        0.0,
                        0.0
                    ],
                    [
                        0.1,
                        0.9,
                        0.0,
                        0.0
                    ]
                ],
                [
                    [
                        0.0,
                        0.0,
                        0.1,
                        0.9
                    ],
                    [
                        0.0,
                        0.0,
                        0.9,
                        0.1
                    ],
                    [
                        0.1,
                        0.9,
                        0.0,
                        0.0
                    ],
                    [
                        0.9,
                        0.1,
                        0.0,
                        0.0
                    ]
                ]
            ],
            "C": [
                0.0,
                0.0,
                0.0,
                3.0
            ],
            "D": [
                0.25,
                0.25,
                0.25,
                0.25
            ]
        },
        "matrix_provenance": {
            "A": {
                "source": "InitialParameterization",
                "shape": [
                    4,
                    4
                ],
                "derived": False
            },
            "B": {
                "source": "InitialParameterization",
                "shape": [
                    4,
                    4,
                    4
                ],
                "derived": False,
                "source_order": "action_previous_state_next_state",
                "canonical_order": "next_state_previous_state_action"
            },
            "C": {
                "source": "InitialParameterization",
                "shape": [
                    4
                ],
                "derived": False
            },
            "D": {
                "source": "InitialParameterization",
                "shape": [
                    4
                ],
                "derived": False
            }
        },
        "state_factors": [
            {
                "name": "s",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Current hidden state distribution",
                "index": 2
            },
            {
                "name": "s_prime",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Next hidden state distribution",
                "index": 3
            }
        ],
        "observation_modalities": [],
        "control_factors": [
            {
                "name": "\u03c0",
                "size": 4,
                "dimensions": [
                    4
                ],
                "type": "float",
                "comment": "Policy (distribution over actions)",
                "index": 0
            },
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Action taken",
                "index": 1
            }
        ],
        "adapter_notes": []
    },
    "matrix_provenance": {
        "A": {
            "source": "InitialParameterization",
            "shape": [
                4,
                4
            ],
            "derived": False
        },
        "B": {
            "source": "InitialParameterization",
            "shape": [
                4,
                4,
                4
            ],
            "derived": False,
            "source_order": "action_previous_state_next_state",
            "canonical_order": "next_state_previous_state_action"
        },
        "C": {
            "source": "InitialParameterization",
            "shape": [
                4
            ],
            "derived": False
        },
        "D": {
            "source": "InitialParameterization",
            "shape": [
                4
            ],
            "derived": False
        }
    },
    "canonical_pomdp_schema": "canonical_pomdp_v1",
    "variables": [
        {
            "name": "B",
            "dimensions": [
                4,
                4,
                4
            ],
            "type": "float",
            "comment": "State transitions given state and action"
        },
        {
            "name": "D",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Prior over initial hidden states"
        },
        {
            "name": "s",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Current hidden state distribution"
        },
        {
            "name": "s_prime",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Next hidden state distribution"
        },
        {
            "name": "o",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Current observation (same as state in MDP)"
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
                4,
                4
            ],
            "type": "float",
            "comment": "Observation model: identity (fully observable)"
        },
        {
            "name": "C",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Log-preferences over observations"
        },
        {
            "name": "\u03c0",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Policy (distribution over actions)"
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
        "G": "ExpectedFreeEnergy",
        "s": "HiddenState",
        "s_prime": "NextHiddenState",
        "o": "Observation",
        "\u03c0": "PolicyVector",
        "u": "Action",
        "t": "Time"
    }
}
    gnn_spec.setdefault("initialparameterization", {})
    if A_data is not None: gnn_spec["initialparameterization"]["A"] = A_data
    if B_data is not None: gnn_spec["initialparameterization"]["B"] = B_data
    if C_data is not None: gnn_spec["initialparameterization"]["C"] = C_data
    if D_data is not None: gnn_spec["initialparameterization"]["D"] = D_data
    if E_data is not None: gnn_spec["initialparameterization"]["E"] = E_data
    gnn_spec.setdefault("model_parameters", {})
    gnn_spec["model_parameters"].setdefault("num_timesteps", 25)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Simple MDP Agent"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Simple MDP Agent")
    logger.info("Output directory: %s", output_dir)

    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            correlation_id="render_generated_script",
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        logger.error("Unexpected error: %s", exc)
        traceback.print_exc()
        return 1

    if success:
        logger.info("Simulation completed successfully")
        logger.info("  framework:    %s", results.get("framework"))
        logger.info("  pymdp ver:    %s", results.get("pymdp_version"))
        logger.info("  backend:      %s", results.get("backend"))
        logger.info("  num_timesteps:%s", results.get("num_timesteps"))
        return 0

    logger.error("Simulation failed: %s", results.get("error", "Unknown error"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
