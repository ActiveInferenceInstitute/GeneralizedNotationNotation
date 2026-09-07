#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Curiosity-Driven Active Inference Agent

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Curiosity-Driven Active Inference Agent
Description:  
Generated:    2026-09-06 12:13:34

State Space:
  - Hidden States: 5
  - Observations:  5
  - Actions:       4

Initial matrices present in GNN spec:
  - A (likelihood):   Present
  - B (transitions):  Present
  - C (preferences):  Present
  - D (state prior):  Present
  - E (policy prior): Present
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
    A_data = [[0.9, 0.025, 0.025, 0.025, 0.025], [0.025, 0.9, 0.025, 0.025, 0.025], [0.025, 0.025, 0.9, 0.025, 0.025], [0.025, 0.025, 0.025, 0.9, 0.025], [0.025, 0.025, 0.025, 0.025, 0.9]]
    B_data = [[[0.9, 0.9, 1.0, 0.9], [0.1, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.1, 0.1, 0.0, 0.0], [0.8, 0.9, 0.9, 0.9], [0.1, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.0, 0.0], [0.8, 0.9, 0.9, 0.9], [0.1, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.0, 0.0], [0.8, 0.9, 0.9, 0.9], [0.1, 0.0, 0.1, 0.0]], [[0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.1], [0.1, 0.1, 0.0, 0.1], [0.9, 1.0, 0.9, 1.0]]]
    C_data = [-2.0, -2.0, -2.0, -2.0, 2.0]
    D_data = [0.2, 0.2, 0.2, 0.2, 0.2]
    E_data = [0.25, 0.25, 0.25, 0.25]

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Curiosity-Driven Active Inference Agent",
    "model_name": "Curiosity-Driven Active Inference Agent",
    "description": "An Active Inference agent with:\n- Explicit epistemic value (information gain / Bayesian surprise) component in G\n- Separate instrumental value (preference satisfaction) component\n- Precision parameter \u03b3 weighting epistemic vs instrumental contributions\n- 5 hidden states, 5 observations, 4 actions in a navigation context\n- Agent is rewarded for reducing posterior uncertainty",
    "gnn_section": "ActInfPOMDP",
    "model_parameters": {
        "num_hidden_states": 5,
        "num_obs": 5,
        "num_actions": 4,
        "num_timesteps": 30,
        "epistemic_weight": 1.0,
        "instrumental_weight": 1.0,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 2,
        "num_modalities": 1,
        "state_factors": [
            {
                "name": "s",
                "size": 5,
                "dimensions": [
                    5,
                    1
                ],
                "type": "float",
                "comment": "Hidden state belief",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_prime",
                "size": 5,
                "dimensions": [
                    5,
                    1
                ],
                "type": "float",
                "comment": "Next hidden state belief",
                "index": 1,
                "role": "bookkeeping"
            }
        ],
        "observation_modalities": [
            {
                "name": "o",
                "size": 5,
                "dimensions": [
                    5,
                    1
                ],
                "type": "float",
                "comment": "Current observation",
                "index": 0,
                "role": "factor"
            }
        ],
        "control_factors": [
            {
                "name": "\u03c0",
                "size": 4,
                "dimensions": [
                    4
                ],
                "type": "float",
                "comment": "Policy distribution over actions",
                "index": 0,
                "role": "bookkeeping"
            },
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Selected action",
                "index": 1,
                "role": "factor"
            }
        ],
        "passive_model": False,
        "simulation_params": {}
    },
    "initialparameterization": {
        "A": [
            [
                0.9,
                0.025,
                0.025,
                0.025,
                0.025
            ],
            [
                0.025,
                0.9,
                0.025,
                0.025,
                0.025
            ],
            [
                0.025,
                0.025,
                0.9,
                0.025,
                0.025
            ],
            [
                0.025,
                0.025,
                0.025,
                0.9,
                0.025
            ],
            [
                0.025,
                0.025,
                0.025,
                0.025,
                0.9
            ]
        ],
        "B": [
            [
                [
                    0.9,
                    0.9,
                    1.0,
                    0.9
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.1,
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.8,
                    0.9,
                    0.9,
                    0.9
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.8,
                    0.9,
                    0.9,
                    0.9
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.1,
                    0.1,
                    0.0,
                    0.0
                ],
                [
                    0.8,
                    0.9,
                    0.9,
                    0.9
                ],
                [
                    0.1,
                    0.0,
                    0.1,
                    0.0
                ]
            ],
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.1
                ],
                [
                    0.1,
                    0.1,
                    0.0,
                    0.1
                ],
                [
                    0.9,
                    1.0,
                    0.9,
                    1.0
                ]
            ]
        ],
        "C": [
            -2.0,
            -2.0,
            -2.0,
            -2.0,
            2.0
        ],
        "D": [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2
        ],
        "E": [
            0.25,
            0.25,
            0.25,
            0.25
        ],
        "\u03b3": [
            1.0
        ]
    },
    "structured_pomdp": {
        "matrices": {
            "A": [
                [
                    0.9,
                    0.025,
                    0.025,
                    0.025,
                    0.025
                ],
                [
                    0.025,
                    0.9,
                    0.025,
                    0.025,
                    0.025
                ],
                [
                    0.025,
                    0.025,
                    0.9,
                    0.025,
                    0.025
                ],
                [
                    0.025,
                    0.025,
                    0.025,
                    0.9,
                    0.025
                ],
                [
                    0.025,
                    0.025,
                    0.025,
                    0.025,
                    0.9
                ]
            ],
            "B": [
                [
                    [
                        0.9,
                        0.9,
                        1.0,
                        0.9
                    ],
                    [
                        0.1,
                        0.0,
                        0.1,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]
                ],
                [
                    [
                        0.1,
                        0.1,
                        0.0,
                        0.0
                    ],
                    [
                        0.8,
                        0.9,
                        0.9,
                        0.9
                    ],
                    [
                        0.1,
                        0.0,
                        0.1,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]
                ],
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.1,
                        0.1,
                        0.0,
                        0.0
                    ],
                    [
                        0.8,
                        0.9,
                        0.9,
                        0.9
                    ],
                    [
                        0.1,
                        0.0,
                        0.1,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]
                ],
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.1,
                        0.1,
                        0.0,
                        0.0
                    ],
                    [
                        0.8,
                        0.9,
                        0.9,
                        0.9
                    ],
                    [
                        0.1,
                        0.0,
                        0.1,
                        0.0
                    ]
                ],
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.1
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.1
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.1
                    ],
                    [
                        0.1,
                        0.1,
                        0.0,
                        0.1
                    ],
                    [
                        0.9,
                        1.0,
                        0.9,
                        1.0
                    ]
                ]
            ],
            "C": [
                -2.0,
                -2.0,
                -2.0,
                -2.0,
                2.0
            ],
            "D": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2
            ],
            "E": [
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
                    5,
                    5
                ],
                "derived": False
            },
            "B": {
                "source": "InitialParameterization",
                "shape": [
                    5,
                    5,
                    4
                ],
                "derived": False,
                "declared_order": [
                    "next_state",
                    "previous_state",
                    "action"
                ],
                "claimed_slice_convention": None,
                "detected_order": [
                    "next_state",
                    "previous_state",
                    "action"
                ],
                "canonical_order": "next_state_previous_state_action",
                "contradiction": False,
                "reason": None,
                "source_order": "next_state_previous_state_action"
            },
            "C": {
                "source": "InitialParameterization",
                "shape": [
                    5
                ],
                "derived": False
            },
            "D": {
                "source": "InitialParameterization",
                "shape": [
                    5
                ],
                "derived": False
            },
            "E": {
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
                "size": 5,
                "dimensions": [
                    5,
                    1
                ],
                "type": "float",
                "comment": "Hidden state belief",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_prime",
                "size": 5,
                "dimensions": [
                    5,
                    1
                ],
                "type": "float",
                "comment": "Next hidden state belief",
                "index": 1,
                "role": "bookkeeping"
            }
        ],
        "observation_modalities": [
            {
                "name": "o",
                "size": 5,
                "dimensions": [
                    5,
                    1
                ],
                "type": "float",
                "comment": "Current observation",
                "index": 0,
                "role": "factor"
            }
        ],
        "control_factors": [
            {
                "name": "\u03c0",
                "size": 4,
                "dimensions": [
                    4
                ],
                "type": "float",
                "comment": "Policy distribution over actions",
                "index": 0,
                "role": "bookkeeping"
            },
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Selected action",
                "index": 1,
                "role": "factor"
            }
        ],
        "adapter_notes": []
    },
    "matrix_provenance": {
        "A": {
            "source": "InitialParameterization",
            "shape": [
                5,
                5
            ],
            "derived": False
        },
        "B": {
            "source": "InitialParameterization",
            "shape": [
                5,
                5,
                4
            ],
            "derived": False,
            "declared_order": [
                "next_state",
                "previous_state",
                "action"
            ],
            "claimed_slice_convention": None,
            "detected_order": [
                "next_state",
                "previous_state",
                "action"
            ],
            "canonical_order": "next_state_previous_state_action",
            "contradiction": False,
            "reason": None,
            "source_order": "next_state_previous_state_action"
        },
        "C": {
            "source": "InitialParameterization",
            "shape": [
                5
            ],
            "derived": False
        },
        "D": {
            "source": "InitialParameterization",
            "shape": [
                5
            ],
            "derived": False
        },
        "E": {
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
            "name": "s",
            "dimensions": [
                5,
                1
            ],
            "type": "float",
            "comment": "Hidden state belief"
        },
        {
            "name": "s_prime",
            "dimensions": [
                5,
                1
            ],
            "type": "float",
            "comment": "Next hidden state belief"
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
                5,
                1
            ],
            "type": "float",
            "comment": "Current observation"
        },
        {
            "name": "\u03c0",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Policy distribution over actions"
        },
        {
            "name": "u",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Selected action"
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
            "source": "C",
            "relation": ">",
            "target": "G_ins"
        },
        {
            "source": "G_epi",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "G_ins",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "\u03b3",
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
        }
    ],
    "ontology_mapping": {
        "A": "LikelihoodMatrix",
        "B": "TransitionMatrix",
        "C": "LogPreferenceVector",
        "D": "PriorOverHiddenStates",
        "E": "Habit",
        "s": "HiddenState",
        "s_prime": "NextHiddenState",
        "o": "Observation",
        "\u03c0": "PolicyVector",
        "u": "Action",
        "G": "ExpectedFreeEnergy",
        "G_epi": "EpistemicValue",
        "G_ins": "InstrumentalValue",
        "\u03b3": "PrecisionParameter",
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

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Curiosity-Driven Active Inference Agent"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Curiosity-Driven Active Inference Agent")
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
