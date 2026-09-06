#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Static Perception Model

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Static Perception Model
Description:  
Generated:    2026-09-05 20:32:38

State Space:
  - Hidden States: 2
  - Observations:  2
  - Actions:       2

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
    A_data = [[0.8181818181818181, 0.11111111111111112], [0.18181818181818182, 0.888888888888889]]
    B_data = [[[0.95, 0.05], [0.05, 0.95]], [[0.05, 0.95], [0.95, 0.05]]]
    C_data = [0.0, 0.0]
    D_data = [0.5, 0.5]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Static Perception Model",
    "model_name": "Static Perception Model",
    "description": "The simplest Active Inference model demonstrating pure perception:\n- 2 hidden states mapped to 2 observations via a recognition matrix A\n- Prior D encodes initial beliefs over hidden states\n- Minimal 2-action transition component B so the model is a complete POMDP\n(renderable and executable by pymdp and the general simulation frameworks)\n- Suitable as a minimal baseline and for testing perception-only inference",
    "gnn_section": "ActInfPOMDP",
    "model_parameters": {
        "num_hidden_states": 2,
        "num_obs": 2,
        "num_actions": 2,
        "num_timesteps": 5,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 1,
        "num_modalities": 1,
        "state_factors": [
            {
                "name": "s",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Hidden state (posterior belief)",
                "index": 0,
                "role": "factor"
            }
        ],
        "observation_modalities": [
            {
                "name": "o",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Observation (one-hot encoded)",
                "index": 0,
                "role": "factor"
            }
        ],
        "control_factors": [
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Action taken",
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
                0.8181818181818181,
                0.11111111111111112
            ],
            [
                0.18181818181818182,
                0.888888888888889
            ]
        ],
        "B": [
            [
                [
                    0.95,
                    0.05
                ],
                [
                    0.05,
                    0.95
                ]
            ],
            [
                [
                    0.05,
                    0.95
                ],
                [
                    0.95,
                    0.05
                ]
            ]
        ],
        "C": [
            0.0,
            0.0
        ],
        "D": [
            0.5,
            0.5
        ]
    },
    "structured_pomdp": {
        "matrices": {
            "A": [
                [
                    0.9,
                    0.1
                ],
                [
                    0.2,
                    0.8
                ]
            ],
            "B": [
                [
                    [
                        0.95,
                        0.05
                    ],
                    [
                        0.05,
                        0.95
                    ]
                ],
                [
                    [
                        0.05,
                        0.95
                    ],
                    [
                        0.95,
                        0.05
                    ]
                ]
            ],
            "C": [
                0.0,
                0.0
            ],
            "D": [
                0.5,
                0.5
            ]
        },
        "matrix_provenance": {
            "A": {
                "source": "InitialParameterization",
                "shape": [
                    2,
                    2
                ],
                "derived": False
            },
            "B": {
                "source": "InitialParameterization",
                "shape": [
                    2,
                    2,
                    2
                ],
                "derived": False,
                "declared_order": [
                    "next_state",
                    "previous_state",
                    "action"
                ],
                "claimed_slice_convention": "rows_next_columns_previous",
                "detected_order": None,
                "canonical_order": "next_state_previous_state_action",
                "contradiction": False,
                "reason": None,
                "source_order": "action_previous_state_next_state"
            },
            "C": {
                "source": "InitialParameterization",
                "shape": [
                    2
                ],
                "derived": False
            },
            "D": {
                "source": "InitialParameterization",
                "shape": [
                    2
                ],
                "derived": False
            }
        },
        "state_factors": [
            {
                "name": "s",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Hidden state (posterior belief)",
                "index": 0,
                "role": "factor"
            }
        ],
        "observation_modalities": [
            {
                "name": "o",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Observation (one-hot encoded)",
                "index": 0,
                "role": "factor"
            }
        ],
        "control_factors": [
            {
                "name": "u",
                "size": 1,
                "dimensions": [
                    1
                ],
                "type": "float",
                "comment": "Action taken",
                "index": 0,
                "role": "factor"
            }
        ],
        "adapter_notes": []
    },
    "matrix_provenance": {
        "A": {
            "source": "InitialParameterization",
            "shape": [
                2,
                2
            ],
            "derived": False
        },
        "B": {
            "source": "InitialParameterization",
            "shape": [
                2,
                2,
                2
            ],
            "derived": False,
            "declared_order": [
                "next_state",
                "previous_state",
                "action"
            ],
            "claimed_slice_convention": "rows_next_columns_previous",
            "detected_order": None,
            "canonical_order": "next_state_previous_state_action",
            "contradiction": False,
            "reason": None,
            "source_order": "action_previous_state_next_state"
        },
        "C": {
            "source": "InitialParameterization",
            "shape": [
                2
            ],
            "derived": False
        },
        "D": {
            "source": "InitialParameterization",
            "shape": [
                2
            ],
            "derived": False
        }
    },
    "canonical_pomdp_schema": "canonical_pomdp_v1",
    "variables": [
        {
            "name": "s",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Hidden state (posterior belief)"
        },
        {
            "name": "o",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Observation (one-hot encoded)"
        },
        {
            "name": "u",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Action taken"
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
            "target": "B"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "u"
        },
        {
            "source": "u",
            "relation": ">",
            "target": "s"
        }
    ],
    "ontology_mapping": {
        "A": "RecognitionMatrix",
        "B": "TransitionMatrix",
        "C": "PreferenceVector",
        "D": "Prior",
        "s": "HiddenState",
        "o": "Observation",
        "u": "Action"
    }
}
    gnn_spec.setdefault("initialparameterization", {})
    if A_data is not None: gnn_spec["initialparameterization"]["A"] = A_data
    if B_data is not None: gnn_spec["initialparameterization"]["B"] = B_data
    if C_data is not None: gnn_spec["initialparameterization"]["C"] = C_data
    if D_data is not None: gnn_spec["initialparameterization"]["D"] = D_data
    if E_data is not None: gnn_spec["initialparameterization"]["E"] = E_data
    gnn_spec.setdefault("model_parameters", {})
    gnn_spec["model_parameters"].setdefault("num_timesteps", 5)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Static Perception Model"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Static Perception Model")
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
