#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Hierarchical Active Inference POMDP

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Hierarchical Active Inference POMDP
Description:  
Generated:    2026-09-08 06:56:54

State Space:
  - Hidden States: 8
  - Observations:  16
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

from gnn.execute.pymdp import execute_pymdp_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run a pymdp 1.0.0 simulation for the GNN model embedded in this file."""
    # Matrices embedded verbatim from the GNN spec.
    A_data = [[0.38249999999999995, 0.042499999999999996, 0.0225, 0.0025, 0.022500000000000003, 0.0025000000000000005, 0.022500000000000003, 0.0025000000000000005], [0.042499999999999996, 0.38249999999999995, 0.0025, 0.0225, 0.0025000000000000005, 0.022500000000000003, 0.0025000000000000005, 0.022500000000000003], [0.21249999999999997, 0.21249999999999997, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.0125, 0.0125], [0.21249999999999997, 0.21249999999999997, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.0125, 0.0125], [0.022500000000000003, 0.0025, 0.38249999999999995, 0.042499999999999996, 0.022500000000000003, 0.0025000000000000005, 0.022500000000000003, 0.0025000000000000005], [0.0025, 0.022500000000000003, 0.042499999999999996, 0.38249999999999995, 0.0025000000000000005, 0.022500000000000003, 0.0025000000000000005, 0.022500000000000003], [0.012499999999999999, 0.012499999999999999, 0.2125, 0.2125, 0.012499999999999999, 0.012499999999999999, 0.0125, 0.0125], [0.012499999999999999, 0.012499999999999999, 0.2125, 0.2125, 0.012499999999999999, 0.012499999999999999, 0.0125, 0.0125], [0.022500000000000003, 0.0025, 0.0225, 0.0025, 0.38249999999999995, 0.0425, 0.022500000000000003, 0.0025000000000000005], [0.0025, 0.022500000000000003, 0.0025, 0.0225, 0.0425, 0.38249999999999995, 0.0025000000000000005, 0.022500000000000003], [0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.21249999999999997, 0.21249999999999997, 0.0125, 0.0125], [0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.21249999999999997, 0.21249999999999997, 0.0125, 0.0125], [0.022500000000000003, 0.0025, 0.0225, 0.0025, 0.022500000000000003, 0.0025000000000000005, 0.3825, 0.0425], [0.0025, 0.022500000000000003, 0.0025, 0.0225, 0.0025000000000000005, 0.022500000000000003, 0.0425, 0.3825], [0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.2125, 0.2125], [0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.012499999999999999, 0.2125, 0.2125]]
    B_data = [[[0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.9], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.9, 0.0], [0.0, 0.1, 0.0], [0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.9], [0.0, 0.0, 0.1]], [[0.0, 0.1, 0.0], [0.0, 0.9, 0.0], [0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.9]], [[0.0, 0.0, 0.9], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.1, 0.0]], [[0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.9, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.9], [0.0, 0.0, 0.1], [0.0, 0.9, 0.0], [0.0, 0.1, 0.0], [0.9, 0.0, 0.0], [0.1, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.9], [0.0, 0.1, 0.0], [0.0, 0.9, 0.0], [0.1, 0.0, 0.0], [0.9, 0.0, 0.0]]]
    C_data = [0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 1.0, 1.5, 1.0, 1.5]
    D_data = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Hierarchical Active Inference POMDP",
    "model_name": "Hierarchical Active Inference POMDP",
    "description": "A two-level hierarchical POMDP where:\n- Level 1 (fast): 4 observations, 4 hidden states, 3 actions\n- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood\n- Higher-level beliefs are updated at a slower timescale\n- Top-down predictions constrain bottom-up inference at Level 1",
    "gnn_section": "ActInfPOMDP_Hierarchical",
    "model_parameters": {
        "num_hidden_states": 8,
        "num_obs": 16,
        "num_actions": 3,
        "num_timesteps": 20,
        "num_hidden_states_l1": 4,
        "num_obs_l1": 4,
        "num_actions_l1": 3,
        "num_context_states_l2": 2,
        "timescale_ratio": 5,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 2,
        "num_modalities": 2,
        "state_factors": [
            {
                "name": "s_level1",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Level 1 hidden state distribution",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_level2",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Level 2 contextual hidden state",
                "index": 3,
                "role": "factor"
            }
        ],
        "observation_modalities": [
            {
                "name": "o_level1",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Level 1 observations",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "o_level2",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Level 2 observation (= Level 1 hidden state distribution)",
                "index": 1,
                "role": "factor"
            }
        ],
        "control_factors": [],
        "passive_model": False,
        "simulation_params": {}
    },
    "initialparameterization": {
        "A": [
            [
                0.38249999999999995,
                0.042499999999999996,
                0.0225,
                0.0025,
                0.022500000000000003,
                0.0025000000000000005,
                0.022500000000000003,
                0.0025000000000000005
            ],
            [
                0.042499999999999996,
                0.38249999999999995,
                0.0025,
                0.0225,
                0.0025000000000000005,
                0.022500000000000003,
                0.0025000000000000005,
                0.022500000000000003
            ],
            [
                0.21249999999999997,
                0.21249999999999997,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.0125,
                0.0125
            ],
            [
                0.21249999999999997,
                0.21249999999999997,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.0125,
                0.0125
            ],
            [
                0.022500000000000003,
                0.0025,
                0.38249999999999995,
                0.042499999999999996,
                0.022500000000000003,
                0.0025000000000000005,
                0.022500000000000003,
                0.0025000000000000005
            ],
            [
                0.0025,
                0.022500000000000003,
                0.042499999999999996,
                0.38249999999999995,
                0.0025000000000000005,
                0.022500000000000003,
                0.0025000000000000005,
                0.022500000000000003
            ],
            [
                0.012499999999999999,
                0.012499999999999999,
                0.2125,
                0.2125,
                0.012499999999999999,
                0.012499999999999999,
                0.0125,
                0.0125
            ],
            [
                0.012499999999999999,
                0.012499999999999999,
                0.2125,
                0.2125,
                0.012499999999999999,
                0.012499999999999999,
                0.0125,
                0.0125
            ],
            [
                0.022500000000000003,
                0.0025,
                0.0225,
                0.0025,
                0.38249999999999995,
                0.0425,
                0.022500000000000003,
                0.0025000000000000005
            ],
            [
                0.0025,
                0.022500000000000003,
                0.0025,
                0.0225,
                0.0425,
                0.38249999999999995,
                0.0025000000000000005,
                0.022500000000000003
            ],
            [
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.21249999999999997,
                0.21249999999999997,
                0.0125,
                0.0125
            ],
            [
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.21249999999999997,
                0.21249999999999997,
                0.0125,
                0.0125
            ],
            [
                0.022500000000000003,
                0.0025,
                0.0225,
                0.0025,
                0.022500000000000003,
                0.0025000000000000005,
                0.3825,
                0.0425
            ],
            [
                0.0025,
                0.022500000000000003,
                0.0025,
                0.0225,
                0.0025000000000000005,
                0.022500000000000003,
                0.0425,
                0.3825
            ],
            [
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.2125,
                0.2125
            ],
            [
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.012499999999999999,
                0.2125,
                0.2125
            ]
        ],
        "B": [
            [
                [
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.1,
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
                    0.1,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.9
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
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.1,
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
                    0.1
                ],
                [
                    0.0,
                    0.0,
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
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.1,
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
                    0.0,
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.1
                ]
            ],
            [
                [
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.9,
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
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.9
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.9
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
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.1,
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
                    0.1,
                    0.0
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
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.1,
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
                    0.9
                ],
                [
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.9,
                    0.0,
                    0.0
                ],
                [
                    0.1,
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
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.9
                ],
                [
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.0,
                    0.9,
                    0.0
                ],
                [
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.9,
                    0.0,
                    0.0
                ]
            ]
        ],
        "C": [
            0.1,
            0.6,
            0.1,
            0.6,
            0.1,
            0.6,
            0.1,
            0.6,
            0.1,
            0.6,
            0.1,
            0.6,
            1.0,
            1.5,
            1.0,
            1.5
        ],
        "D": [
            0.125,
            0.125,
            0.125,
            0.125,
            0.125,
            0.125,
            0.125,
            0.125
        ]
    },
    "structured_pomdp": {
        "matrices": {
            "A_level1": [
                [
                    0.85,
                    0.05,
                    0.05,
                    0.05
                ],
                [
                    0.05,
                    0.85,
                    0.05,
                    0.05
                ],
                [
                    0.05,
                    0.05,
                    0.85,
                    0.05
                ],
                [
                    0.05,
                    0.05,
                    0.05,
                    0.85
                ]
            ],
            "B_level1": [
                [
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
                [
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        0.0
                    ]
                ],
                [
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
                    ],
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
                    ]
                ]
            ],
            "C_level1": [
                0.1,
                0.1,
                0.1,
                1.0
            ],
            "D_level1": [
                0.25,
                0.25,
                0.25,
                0.25
            ],
            "A_level2": [
                [
                    0.9,
                    0.1
                ],
                [
                    0.1,
                    0.9
                ],
                [
                    0.5,
                    0.5
                ],
                [
                    0.5,
                    0.5
                ]
            ],
            "B_level2": [
                [
                    0.9,
                    0.1
                ],
                [
                    0.1,
                    0.9
                ]
            ],
            "C_level2": [
                0.0,
                0.5,
                0.0,
                0.5
            ],
            "D_level2": [
                0.5,
                0.5
            ]
        },
        "matrix_provenance": {
            "A_level1": {
                "source": "InitialParameterization",
                "shape": [
                    4,
                    4
                ],
                "derived": False
            },
            "B_level1": {
                "source": "InitialParameterization",
                "shape": [
                    3,
                    4,
                    4
                ],
                "derived": False
            },
            "C_level1": {
                "source": "InitialParameterization",
                "shape": [
                    4
                ],
                "derived": False
            },
            "D_level1": {
                "source": "InitialParameterization",
                "shape": [
                    4
                ],
                "derived": False
            },
            "A_level2": {
                "source": "InitialParameterization",
                "shape": [
                    4,
                    2
                ],
                "derived": False
            },
            "B_level2": {
                "source": "InitialParameterization",
                "shape": [
                    2,
                    2
                ],
                "derived": False
            },
            "C_level2": {
                "source": "InitialParameterization",
                "shape": [
                    4
                ],
                "derived": False
            },
            "D_level2": {
                "source": "InitialParameterization",
                "shape": [
                    2
                ],
                "derived": False
            },
            "A": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "A_level1",
                    "A_level2"
                ],
                "shape": [
                    16,
                    8
                ],
                "derived": True
            },
            "B": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "B_level1",
                    "B_level2"
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
                    "C_level1",
                    "C_level2"
                ],
                "shape": [
                    16
                ],
                "derived": True
            },
            "D": {
                "source": "factored_joint_composition",
                "source_keys": [
                    "D_level1",
                    "D_level2"
                ],
                "shape": [
                    8
                ],
                "derived": True
            }
        },
        "state_factors": [
            {
                "name": "s_level1",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Level 1 hidden state distribution",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_level2",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Level 2 contextual hidden state",
                "index": 3,
                "role": "factor"
            }
        ],
        "observation_modalities": [
            {
                "name": "o_level1",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Level 1 observations",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "o_level2",
                "size": 4,
                "dimensions": [
                    4,
                    1
                ],
                "type": "float",
                "comment": "Level 2 observation (= Level 1 hidden state distribution)",
                "index": 1,
                "role": "factor"
            }
        ],
        "control_factors": [],
        "adapter_notes": []
    },
    "matrix_provenance": {
        "A_level1": {
            "source": "InitialParameterization",
            "shape": [
                4,
                4
            ],
            "derived": False
        },
        "B_level1": {
            "source": "InitialParameterization",
            "shape": [
                3,
                4,
                4
            ],
            "derived": False
        },
        "C_level1": {
            "source": "InitialParameterization",
            "shape": [
                4
            ],
            "derived": False
        },
        "D_level1": {
            "source": "InitialParameterization",
            "shape": [
                4
            ],
            "derived": False
        },
        "A_level2": {
            "source": "InitialParameterization",
            "shape": [
                4,
                2
            ],
            "derived": False
        },
        "B_level2": {
            "source": "InitialParameterization",
            "shape": [
                2,
                2
            ],
            "derived": False
        },
        "C_level2": {
            "source": "InitialParameterization",
            "shape": [
                4
            ],
            "derived": False
        },
        "D_level2": {
            "source": "InitialParameterization",
            "shape": [
                2
            ],
            "derived": False
        },
        "A": {
            "source": "factored_joint_composition",
            "source_keys": [
                "A_level1",
                "A_level2"
            ],
            "shape": [
                16,
                8
            ],
            "derived": True
        },
        "B": {
            "source": "factored_joint_composition",
            "source_keys": [
                "B_level1",
                "B_level2"
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
                "C_level1",
                "C_level2"
            ],
            "shape": [
                16
            ],
            "derived": True
        },
        "D": {
            "source": "factored_joint_composition",
            "source_keys": [
                "D_level1",
                "D_level2"
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
            "name": "s_level1",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Level 1 hidden state distribution"
        },
        {
            "name": "x_next1",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Level 1 next hidden state"
        },
        {
            "name": "G1",
            "dimensions": [
                "\u03c01"
            ],
            "type": "float",
            "comment": "Level 1 Expected Free Energy"
        },
        {
            "name": "s_level2",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Level 2 contextual hidden state"
        },
        {
            "name": "G2",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Level 2 Expected Free Energy"
        },
        {
            "name": "t1",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Fast timescale counter"
        },
        {
            "name": "t2",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Slow timescale counter"
        },
        {
            "name": "o_level1",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Level 1 observations"
        },
        {
            "name": "o_level2",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Level 2 observation (= Level 1 hidden state distribution)"
        },
        {
            "name": "\u03c01",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Level 1 policy (actions)"
        },
        {
            "name": "u_level1",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Level 1 action"
        }
    ],
    "connections": [
        {
            "source": "D_level1",
            "relation": ">",
            "target": "s_level1"
        },
        {
            "source": "s_level1",
            "relation": "-",
            "target": "A_level1"
        },
        {
            "source": "s_level1",
            "relation": ">",
            "target": "x_next1"
        },
        {
            "source": "A_level1",
            "relation": "-",
            "target": "o_level1"
        },
        {
            "source": "C_level1",
            "relation": ">",
            "target": "G1"
        },
        {
            "source": "G1",
            "relation": ">",
            "target": "\u03c01"
        },
        {
            "source": "\u03c01",
            "relation": ">",
            "target": "u_level1"
        },
        {
            "source": "B_level1",
            "relation": ">",
            "target": "u_level1"
        },
        {
            "source": "u_level1",
            "relation": ">",
            "target": "x_next1"
        },
        {
            "source": "s_level1",
            "relation": ">",
            "target": "o_level2"
        },
        {
            "source": "D_level2",
            "relation": ">",
            "target": "s_level2"
        },
        {
            "source": "s_level2",
            "relation": "-",
            "target": "A_level2"
        },
        {
            "source": "A_level2",
            "relation": ">",
            "target": "D_level1"
        },
        {
            "source": "s_level2",
            "relation": "-",
            "target": "B_level2"
        },
        {
            "source": "C_level2",
            "relation": ">",
            "target": "G2"
        },
        {
            "source": "G2",
            "relation": ">",
            "target": "s_level2"
        }
    ],
    "ontology_mapping": {
        "A_level1": "LikelihoodMatrix",
        "B_level1": "TransitionMatrix",
        "C_level1": "LogPreferenceVector",
        "D_level1": "PriorOverHiddenStates",
        "s_level1": "HiddenState",
        "o_level1": "Observation",
        "\u03c01": "PolicyVector",
        "u_level1": "Action",
        "G1": "ExpectedFreeEnergy",
        "A_level2": "HigherLevelLikelihoodMatrix",
        "B_level2": "ContextTransitionMatrix",
        "s_level2": "ContextualHiddenState",
        "o_level2": "HigherLevelObservation",
        "G2": "HigherLevelExpectedFreeEnergy"
    }
}
    gnn_spec.setdefault("initialparameterization", {})
    if A_data is not None: gnn_spec["initialparameterization"]["A"] = A_data
    if B_data is not None: gnn_spec["initialparameterization"]["B"] = B_data
    if C_data is not None: gnn_spec["initialparameterization"]["C"] = C_data
    if D_data is not None: gnn_spec["initialparameterization"]["D"] = D_data
    if E_data is not None: gnn_spec["initialparameterization"]["E"] = E_data
    gnn_spec.setdefault("model_parameters", {})
    gnn_spec["model_parameters"].setdefault("num_timesteps", 20)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Hierarchical Active Inference POMDP"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Hierarchical Active Inference POMDP")
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
