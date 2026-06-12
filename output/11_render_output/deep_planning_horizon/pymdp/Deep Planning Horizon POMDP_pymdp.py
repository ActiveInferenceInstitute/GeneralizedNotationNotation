#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Deep Planning Horizon POMDP

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Deep Planning Horizon POMDP
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
    A_data = [[0.9, 0.05, 0.025, 0.025], [0.05, 0.9, 0.025, 0.025], [0.025, 0.025, 0.9, 0.05], [0.025, 0.025, 0.05, 0.9]]
    B_data = [[[0.9, 0.9, 0.8, 0.7000000000000001], [0.0, 0.1, 0.0, 0.10000000000000002], [0.0, 0.0, 0.1, 0.10000000000000002], [0.1, 0.0, 0.1, 0.1]], [[0.1, 0.0, 0.1, 0.10000000000000002], [0.9, 0.9, 0.8, 0.7000000000000001], [0.0, 0.1, 0.0, 0.10000000000000002], [0.0, 0.0, 0.1, 0.1]], [[0.0, 0.0, 0.1, 0.10000000000000002], [0.1, 0.0, 0.1, 0.10000000000000002], [0.9, 0.9, 0.8, 0.7000000000000001], [0.0, 0.1, 0.0, 0.1]], [[0.0, 0.1, 0.0, 0.10000000000000002], [0.0, 0.0, 0.1, 0.10000000000000002], [0.1, 0.0, 0.1, 0.10000000000000002], [0.9, 0.9, 0.8, 0.7]]]
    C_data = [-1.0, -0.5, -0.5, 2.0]
    D_data = [0.25, 0.25, 0.25, 0.25]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Deep Planning Horizon POMDP",
    "model_name": "Deep Planning Horizon POMDP",
    "description": "An Active Inference POMDP with deep (T=5) planning horizon:\n- Evaluates policies over 5 future timesteps before acting\n- Uses rollout Expected Free Energy accumulation\n- 4 hidden states, 4 observations, 4 actions\n- Each action policy is a sequence of T actions: \u03c0 = [a_1, a_2, ..., a_T]\n- Enables sophisticated multi-step reasoning and delayed reward attribution",
    "model_parameters": {
        "num_hidden_states": 4,
        "num_obs": 4,
        "num_actions": 4,
        "planning_horizon": 5,
        "num_policies": 64,
        "num_timesteps": 30,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 6,
        "num_modalities": 1,
        "state_factors": [
            {
                "name": "s",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Current hidden state belief",
                "index": 1
            },
            {
                "name": "s_tau1",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=1",
                "index": 2
            },
            {
                "name": "s_tau2",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=2",
                "index": 3
            },
            {
                "name": "s_tau3",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=3",
                "index": 4
            },
            {
                "name": "s_tau4",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=4",
                "index": 5
            },
            {
                "name": "s_tau5",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=5",
                "index": 6
            }
        ],
        "observation_modalities": [
            {
                "name": "o",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Current observation",
                "index": 1
            }
        ],
        "control_factors": [
            {
                "name": "\u03c0",
                "size": 64,
                "dimensions": [
                    64
                ],
                "type": "float",
                "comment": "Policy distribution (over T-step action sequences)",
                "index": 1
            },
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Selected first action from best policy",
                "index": 2
            }
        ],
        "passive_model": False,
        "simulation_params": {}
    },
    "initialparameterization": {
        "A": [
            [
                0.9,
                0.05,
                0.025,
                0.025
            ],
            [
                0.05,
                0.9,
                0.025,
                0.025
            ],
            [
                0.025,
                0.025,
                0.9,
                0.05
            ],
            [
                0.025,
                0.025,
                0.05,
                0.9
            ]
        ],
        "B": [
            [
                [
                    0.9,
                    0.9,
                    0.8,
                    0.7000000000000001
                ],
                [
                    0.0,
                    0.1,
                    0.0,
                    0.10000000000000002
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                    0.10000000000000002
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.1
                ]
            ],
            [
                [
                    0.1,
                    0.0,
                    0.1,
                    0.10000000000000002
                ],
                [
                    0.9,
                    0.9,
                    0.8,
                    0.7000000000000001
                ],
                [
                    0.0,
                    0.1,
                    0.0,
                    0.10000000000000002
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                    0.1
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.1,
                    0.10000000000000002
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.10000000000000002
                ],
                [
                    0.9,
                    0.9,
                    0.8,
                    0.7000000000000001
                ],
                [
                    0.0,
                    0.1,
                    0.0,
                    0.1
                ]
            ],
            [
                [
                    0.0,
                    0.1,
                    0.0,
                    0.10000000000000002
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                    0.10000000000000002
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.10000000000000002
                ],
                [
                    0.9,
                    0.9,
                    0.8,
                    0.7
                ]
            ]
        ],
        "C": [
            -1.0,
            -0.5,
            -0.5,
            2.0
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
                    0.9,
                    0.05,
                    0.025,
                    0.025
                ],
                [
                    0.05,
                    0.9,
                    0.025,
                    0.025
                ],
                [
                    0.025,
                    0.025,
                    0.9,
                    0.05
                ],
                [
                    0.025,
                    0.025,
                    0.05,
                    0.9
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
                        0.0,
                        0.9,
                        0.1,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.9,
                        0.1
                    ],
                    [
                        0.1,
                        0.0,
                        0.0,
                        0.9
                    ]
                ],
                [
                    [
                        0.9,
                        0.0,
                        0.0,
                        0.1
                    ],
                    [
                        0.1,
                        0.9,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.1,
                        0.9,
                        0.0
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
                        0.8,
                        0.1,
                        0.1,
                        0.0
                    ],
                    [
                        0.0,
                        0.8,
                        0.1,
                        0.1
                    ],
                    [
                        0.1,
                        0.0,
                        0.8,
                        0.1
                    ],
                    [
                        0.1,
                        0.1,
                        0.0,
                        0.8
                    ]
                ],
                [
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
                    ]
                ]
            ],
            "C": [
                -1.0,
                -0.5,
                -0.5,
                2.0
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
                "comment": "Current hidden state belief",
                "index": 1
            },
            {
                "name": "s_tau1",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=1",
                "index": 2
            },
            {
                "name": "s_tau2",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=2",
                "index": 3
            },
            {
                "name": "s_tau3",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=3",
                "index": 4
            },
            {
                "name": "s_tau4",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=4",
                "index": 5
            },
            {
                "name": "s_tau5",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Predicted state at tau=5",
                "index": 6
            }
        ],
        "observation_modalities": [
            {
                "name": "o",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Current observation",
                "index": 1
            }
        ],
        "control_factors": [
            {
                "name": "\u03c0",
                "size": 64,
                "dimensions": [
                    64
                ],
                "type": "float",
                "comment": "Policy distribution (over T-step action sequences)",
                "index": 1
            },
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Selected first action from best policy",
                "index": 2
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
            "name": "D",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Prior over initial states"
        },
        {
            "name": "s",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Current hidden state belief"
        },
        {
            "name": "s_tau1",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Predicted state at tau=1"
        },
        {
            "name": "s_tau2",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Predicted state at tau=2"
        },
        {
            "name": "s_tau3",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Predicted state at tau=3"
        },
        {
            "name": "s_tau4",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Predicted state at tau=4"
        },
        {
            "name": "s_tau5",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Predicted state at tau=5"
        },
        {
            "name": "F",
            "dimensions": [
                "\u03c0"
            ],
            "type": "float",
            "comment": "Variational Free Energy for current state"
        },
        {
            "name": "C",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Preferences (per observation)"
        },
        {
            "name": "o",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Current observation"
        },
        {
            "name": "B",
            "dimensions": [
                4,
                4,
                4
            ],
            "type": "float",
            "comment": "Transition matrix (4 actions)"
        },
        {
            "name": "\u03c0",
            "dimensions": [
                64
            ],
            "type": "float",
            "comment": "Policy distribution (over T-step action sequences)"
        },
        {
            "name": "u",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Selected first action from best policy"
        },
        {
            "name": "t",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Discrete time step (action timestep)"
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
            "relation": "-",
            "target": "F"
        },
        {
            "source": "o",
            "relation": "-",
            "target": "F"
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
            "source": "s",
            "relation": ">",
            "target": "s_tau1"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "s_tau1"
        },
        {
            "source": "s_tau1",
            "relation": ">",
            "target": "s_tau2"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "s_tau2"
        },
        {
            "source": "s_tau2",
            "relation": ">",
            "target": "s_tau3"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "s_tau3"
        },
        {
            "source": "s_tau3",
            "relation": ">",
            "target": "s_tau4"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "s_tau4"
        },
        {
            "source": "s_tau4",
            "relation": ">",
            "target": "s_tau5"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "s_tau1"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "s_tau2"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "s_tau3"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "s_tau4"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "s_tau5"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G_tau1"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G_tau2"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G_tau3"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G_tau4"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G_tau5"
        },
        {
            "source": "G_tau1",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "G_tau2",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "G_tau3",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "G_tau4",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "G_tau5",
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
        }
    ],
    "ontology_mapping": {
        "A": "LikelihoodMatrix",
        "B": "TransitionMatrix",
        "C": "LogPreferenceVector",
        "D": "PriorOverHiddenStates",
        "E": "PolicyPrior",
        "s": "HiddenState",
        "o": "Observation",
        "\u03c0": "PolicySequenceDistribution",
        "u": "Action",
        "G": "CumulativeExpectedFreeEnergy",
        "F": "VariationalFreeEnergy",
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
    gnn_spec["model_parameters"].setdefault("num_timesteps", 30)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Deep Planning Horizon POMDP"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Deep Planning Horizon POMDP")
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
