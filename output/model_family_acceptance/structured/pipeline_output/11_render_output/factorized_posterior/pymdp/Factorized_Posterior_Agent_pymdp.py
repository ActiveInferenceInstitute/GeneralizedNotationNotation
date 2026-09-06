#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Factorized Posterior Agent

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Factorized Posterior Agent
Description:  
Generated:    2026-09-06 11:31:09

State Space:
  - Hidden States: 8
  - Observations:  6
  - Actions:       3

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
    A_data = [[0.63, 0.09000000000000001, 0.010000000000000002, 0.06999999999999999, 0.010000000000000002, 0.010000000000000002, 0.010000000000000002, 0.010000000000000002], [0.06999999999999999, 0.010000000000000002, 0.09000000000000001, 0.63, 0.09000000000000001, 0.09000000000000001, 0.09000000000000001, 0.09000000000000001], [0.09000000000000001, 0.63, 0.06999999999999999, 0.010000000000000002, 0.010000000000000002, 0.010000000000000002, 0.010000000000000002, 0.010000000000000002], [0.010000000000000002, 0.06999999999999999, 0.63, 0.09000000000000001, 0.09000000000000001, 0.09000000000000001, 0.09000000000000001, 0.09000000000000001], [0.18000000000000002, 0.18000000000000002, 0.020000000000000004, 0.020000000000000004, 0.08000000000000002, 0.08000000000000002, 0.08000000000000002, 0.08000000000000002], [0.020000000000000004, 0.020000000000000004, 0.18000000000000002, 0.18000000000000002, 0.7200000000000001, 0.7200000000000001, 0.7200000000000001, 0.7200000000000001]]
    B_data = [[[0.9, 0.1, 0.9], [0.0, 0.0, 0.0], [0.1, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0], [0.1, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.9, 0.0]], [[0.1, 0.9, 0.0], [0.0, 0.0, 0.0], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 0.0], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0], [0.1, 0.0, 0.1], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0], [0.1, 0.0, 0.1]], [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 0.0], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 0.0], [0.9, 0.1, 0.9]]]
    C_data = [0.5, 0.5, 0.5, 0.5, 1.5, 1.5]
    D_data = [0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Factorized Posterior Agent",
    "model_name": "Factorized Posterior Agent",
    "description": "A mean-field factorized POMDP agent. The joint posterior over two\nindependent state factors `s_1` (location) and `s_2` (goal identity) is\napproximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).\nThis is the canonical simplification used in variational inference when\nexact joint posteriors are computationally intractable.\n- Two state factors: location (4 states), goal (2 states)\n- Two observation modalities: visual (3 obs), proprioceptive (2 obs)\n- Separate transition matrices B_1 (location \u00d7 action) and B_2 (goal is static)\n- Explicit factorization declared in ## Equations\n- Tests multi-factor / multi-modality handling in the parser",
    "gnn_section": "ActInfFactorized",
    "model_parameters": {
        "num_hidden_states": 8,
        "num_obs": 6,
        "num_hidden_states_factor0": 4,
        "num_hidden_states_factor1": 2,
        "num_obs_modality0": 3,
        "num_obs_modality1": 2,
        "num_actions": 3,
        "num_factors": 2,
        "num_modalities": 2,
        "num_timesteps": 15,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 2,
        "state_factors": [
            {
                "name": "s_f0",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Factor 0: agent location (4 possible positions)",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_f1",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Factor 1: goal identity (2 possible goals)",
                "index": 1,
                "role": "factor"
            }
        ],
        "observation_modalities": [
            {
                "name": "o_m0",
                "size": 3,
                "dimensions": [
                    3,
                    1
                ],
                "type": "float",
                "comment": "Modality 0: visual observation (3 visual cues)",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "o_m1",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Modality 1: proprioceptive observation (2 body states)",
                "index": 1,
                "role": "factor"
            }
        ],
        "control_factors": [
            {
                "name": "u",
                "size": 3,
                "dimensions": [
                    3,
                    1
                ],
                "type": "float",
                "comment": "3 possible actions: stay, forward, backward",
                "index": 0,
                "role": "factor"
            }
        ],
        "passive_model": False,
        "simulation_params": {}
    },
    "initialparameterization": {
        "A": [
            [
                0.63,
                0.09000000000000001,
                0.010000000000000002,
                0.06999999999999999,
                0.010000000000000002,
                0.010000000000000002,
                0.010000000000000002,
                0.010000000000000002
            ],
            [
                0.06999999999999999,
                0.010000000000000002,
                0.09000000000000001,
                0.63,
                0.09000000000000001,
                0.09000000000000001,
                0.09000000000000001,
                0.09000000000000001
            ],
            [
                0.09000000000000001,
                0.63,
                0.06999999999999999,
                0.010000000000000002,
                0.010000000000000002,
                0.010000000000000002,
                0.010000000000000002,
                0.010000000000000002
            ],
            [
                0.010000000000000002,
                0.06999999999999999,
                0.63,
                0.09000000000000001,
                0.09000000000000001,
                0.09000000000000001,
                0.09000000000000001,
                0.09000000000000001
            ],
            [
                0.18000000000000002,
                0.18000000000000002,
                0.020000000000000004,
                0.020000000000000004,
                0.08000000000000002,
                0.08000000000000002,
                0.08000000000000002,
                0.08000000000000002
            ],
            [
                0.020000000000000004,
                0.020000000000000004,
                0.18000000000000002,
                0.18000000000000002,
                0.7200000000000001,
                0.7200000000000001,
                0.7200000000000001,
                0.7200000000000001
            ]
        ],
        "B": [
            [
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ]
            ],
            [
                [
                    0.1,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.1
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.1,
                    0.9
                ]
            ]
        ],
        "C": [
            0.5,
            0.5,
            0.5,
            0.5,
            1.5,
            1.5
        ],
        "D": [
            0.15,
            0.1,
            0.15,
            0.1,
            0.15,
            0.1,
            0.15,
            0.1
        ]
    },
    "structured_pomdp": {
        "matrices": {
            "A_m0": [
                [
                    [
                        0.7,
                        0.1
                    ],
                    [
                        0.1,
                        0.7
                    ],
                    [
                        0.1,
                        0.1
                    ],
                    [
                        0.1,
                        0.1
                    ]
                ],
                [
                    [
                        0.1,
                        0.7
                    ],
                    [
                        0.7,
                        0.1
                    ],
                    [
                        0.1,
                        0.1
                    ],
                    [
                        0.1,
                        0.1
                    ]
                ],
                [
                    [
                        0.2,
                        0.2
                    ],
                    [
                        0.2,
                        0.2
                    ],
                    [
                        0.8,
                        0.8
                    ],
                    [
                        0.8,
                        0.8
                    ]
                ]
            ],
            "A_m1": [
                [
                    0.9,
                    0.1,
                    0.1,
                    0.1
                ],
                [
                    0.1,
                    0.9,
                    0.9,
                    0.9
                ]
            ],
            "D_f0": [
                0.25,
                0.25,
                0.25,
                0.25
            ],
            "D_f1": [
                0.6,
                0.4
            ],
            "C_m0": [
                0.0,
                0.0,
                1.0
            ],
            "C_m1": [
                0.5,
                0.5
            ],
            "B_f0": [
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
                    ],
                    [
                        0.9,
                        0.0,
                        0.0,
                        0.1
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
                ]
            ],
            "B_f1": [
                [
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    1.0
                ]
            ]
        },
        "matrix_provenance": {
            "A_m0": {
                "source": "InitialParameterization",
                "shape": [
                    3,
                    4,
                    2
                ],
                "derived": False
            },
            "A_m1": {
                "source": "InitialParameterization",
                "shape": [
                    2,
                    4
                ],
                "derived": False
            },
            "D_f0": {
                "source": "InitialParameterization",
                "shape": [
                    4
                ],
                "derived": False
            },
            "D_f1": {
                "source": "InitialParameterization",
                "shape": [
                    2
                ],
                "derived": False
            },
            "C_m0": {
                "source": "InitialParameterization",
                "shape": [
                    3
                ],
                "derived": False
            },
            "C_m1": {
                "source": "InitialParameterization",
                "shape": [
                    2
                ],
                "derived": False
            },
            "B_f0": {
                "source": "InitialParameterization",
                "shape": [
                    3,
                    4,
                    4
                ],
                "derived": False
            },
            "B_f1": {
                "source": "InitialParameterization",
                "shape": [
                    2,
                    2
                ],
                "derived": False
            },
            "A": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "A_m0",
                    "A_m1"
                ],
                "shape": [
                    6,
                    8
                ],
                "derived": True
            },
            "B": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "B_f0",
                    "B_f1"
                ],
                "shape": [
                    8,
                    8,
                    3
                ],
                "derived": True,
                "factor_action_counts": [
                    3,
                    1
                ],
                "kronecker_factorized": False,
                "source_order": "next_state_previous_state_action",
                "canonical_order": "next_state_previous_state_action"
            },
            "C": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "C_m0",
                    "C_m1"
                ],
                "shape": [
                    6
                ],
                "derived": True
            },
            "D": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "D_f0",
                    "D_f1"
                ],
                "shape": [
                    8
                ],
                "derived": True
            }
        },
        "state_factors": [
            {
                "name": "s_f0",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Factor 0: agent location (4 possible positions)",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_f1",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Factor 1: goal identity (2 possible goals)",
                "index": 1,
                "role": "factor"
            }
        ],
        "observation_modalities": [
            {
                "name": "o_m0",
                "size": 3,
                "dimensions": [
                    3,
                    1
                ],
                "type": "float",
                "comment": "Modality 0: visual observation (3 visual cues)",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "o_m1",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Modality 1: proprioceptive observation (2 body states)",
                "index": 1,
                "role": "factor"
            }
        ],
        "control_factors": [
            {
                "name": "u",
                "size": 3,
                "dimensions": [
                    3,
                    1
                ],
                "type": "float",
                "comment": "3 possible actions: stay, forward, backward",
                "index": 0,
                "role": "factor"
            }
        ],
        "adapter_notes": []
    },
    "matrix_provenance": {
        "A_m0": {
            "source": "InitialParameterization",
            "shape": [
                3,
                4,
                2
            ],
            "derived": False
        },
        "A_m1": {
            "source": "InitialParameterization",
            "shape": [
                2,
                4
            ],
            "derived": False
        },
        "D_f0": {
            "source": "InitialParameterization",
            "shape": [
                4
            ],
            "derived": False
        },
        "D_f1": {
            "source": "InitialParameterization",
            "shape": [
                2
            ],
            "derived": False
        },
        "C_m0": {
            "source": "InitialParameterization",
            "shape": [
                3
            ],
            "derived": False
        },
        "C_m1": {
            "source": "InitialParameterization",
            "shape": [
                2
            ],
            "derived": False
        },
        "B_f0": {
            "source": "InitialParameterization",
            "shape": [
                3,
                4,
                4
            ],
            "derived": False
        },
        "B_f1": {
            "source": "InitialParameterization",
            "shape": [
                2,
                2
            ],
            "derived": False
        },
        "A": {
            "source": "factored_joint_composition",
            "source_keys": [
                "A_m0",
                "A_m1"
            ],
            "shape": [
                6,
                8
            ],
            "derived": True
        },
        "B": {
            "source": "factored_joint_composition",
            "source_keys": [
                "B_f0",
                "B_f1"
            ],
            "shape": [
                8,
                8,
                3
            ],
            "derived": True,
            "factor_action_counts": [
                3,
                1
            ],
            "kronecker_factorized": False,
            "source_order": "next_state_previous_state_action",
            "canonical_order": "next_state_previous_state_action"
        },
        "C": {
            "source": "factored_joint_composition",
            "source_keys": [
                "C_m0",
                "C_m1"
            ],
            "shape": [
                6
            ],
            "derived": True
        },
        "D": {
            "source": "factored_joint_composition",
            "source_keys": [
                "D_f0",
                "D_f1"
            ],
            "shape": [
                8
            ],
            "derived": True
        }
    },
    "canonical_pomdp_schema": "canonical_pomdp_v1",
    "variables": [
        {
            "name": "s_f0",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Factor 0: agent location (4 possible positions)"
        },
        {
            "name": "s_f1",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Factor 1: goal identity (2 possible goals)"
        },
        {
            "name": "o_m0",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Modality 0: visual observation (3 visual cues)"
        },
        {
            "name": "o_m1",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Modality 1: proprioceptive observation (2 body states)"
        },
        {
            "name": "u",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "3 possible actions: stay, forward, backward"
        }
    ],
    "connections": [
        {
            "source": "D_f0",
            "relation": ">",
            "target": "s_f0"
        },
        {
            "source": "D_f1",
            "relation": ">",
            "target": "s_f1"
        },
        {
            "source": "(s_f0, u)",
            "relation": ">",
            "target": "B_f0"
        },
        {
            "source": "B_f0",
            "relation": ">",
            "target": "s_f0"
        },
        {
            "source": "s_f1",
            "relation": ">",
            "target": "B_f1"
        },
        {
            "source": "B_f1",
            "relation": ">",
            "target": "s_f1"
        },
        {
            "source": "(s_f0, s_f1)",
            "relation": ">",
            "target": "A_m0"
        },
        {
            "source": "A_m0",
            "relation": ">",
            "target": "o_m0"
        },
        {
            "source": "s_f0",
            "relation": ">",
            "target": "A_m1"
        },
        {
            "source": "A_m1",
            "relation": ">",
            "target": "o_m1"
        },
        {
            "source": "C_m0",
            "relation": "-",
            "target": "o_m0"
        },
        {
            "source": "C_m1",
            "relation": "-",
            "target": "o_m1"
        }
    ],
    "ontology_mapping": {
        "s_f0": "HiddenStateFactor0",
        "s_f1": "HiddenStateFactor1",
        "o_m0": "ObservationModality0",
        "o_m1": "ObservationModality1",
        "u": "Action",
        "A_m0": "LikelihoodMatrixModality0",
        "A_m1": "LikelihoodMatrixModality1",
        "D_f0": "PriorFactor0",
        "D_f1": "PriorFactor1",
        "C_m0": "PreferenceModality0",
        "C_m1": "PreferenceModality1"
    }
}
    gnn_spec.setdefault("initialparameterization", {})
    if A_data is not None: gnn_spec["initialparameterization"]["A"] = A_data
    if B_data is not None: gnn_spec["initialparameterization"]["B"] = B_data
    if C_data is not None: gnn_spec["initialparameterization"]["C"] = C_data
    if D_data is not None: gnn_spec["initialparameterization"]["D"] = D_data
    if E_data is not None: gnn_spec["initialparameterization"]["E"] = E_data
    gnn_spec.setdefault("model_parameters", {})
    gnn_spec["model_parameters"].setdefault("num_timesteps", 15)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Factorized Posterior Agent"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Factorized Posterior Agent")
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
