#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Dynamic Perception Model

This file was generated from a GNN specification by
``render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``execute.pymdp.run_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Dynamic Perception Model
Description:  
Generated:    2026-09-06 11:45:16

State Space:
  - Hidden States: 2
  - Observations:  2
  - Actions:       1

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
    A_data = [[0.8181818181818181, 0.11111111111111112], [0.18181818181818182, 0.888888888888889]]
    B_data = [[[0.7], [0.3]], [[0.3], [0.7]]]
    C_data = [0.0, 0.0]
    D_data = [0.5, 0.5]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Dynamic Perception Model",
    "model_name": "Dynamic Perception Model",
    "description": "A dynamic perception model extending the static model with temporal dynamics:\n- 2 hidden states evolving over discrete time via transition matrix B\n- 2 observations generated from states via recognition matrix A\n- Prior D constrains the initial hidden state\n- No action selection \u2014 the agent passively observes a changing world\n- Demonstrates belief updating (state inference) across time steps\n- Suitable for tracking hidden sources from noisy observations",
    "gnn_section": "ActiveInferencePerception",
    "model_parameters": {
        "num_hidden_states": 2,
        "num_obs": 2,
        "num_timesteps": 10,
        "num_actions": 1,
        "b_tensor_order": "next_state_previous_state_action",
        "num_state_factors": 2,
        "num_modalities": 1,
        "state_factors": [
            {
                "name": "s_t",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Hidden state belief at time t",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_prime",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Hidden state belief at time t+1",
                "index": 1,
                "role": "bookkeeping"
            }
        ],
        "observation_modalities": [
            {
                "name": "o_t",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Observation at time t",
                "index": 0,
                "role": "factor"
            }
        ],
        "control_factors": [],
        "passive_model": True,
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
                    0.7
                ],
                [
                    0.3
                ]
            ],
            [
                [
                    0.3
                ],
                [
                    0.7
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
                    0.7,
                    0.3
                ],
                [
                    0.3,
                    0.7
                ]
            ],
            "D": [
                0.5,
                0.5
            ],
            "C": [
                0.0,
                0.0
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
                    1
                ],
                "derived": False,
                "declared_order": [
                    "next_state",
                    "previous_state",
                    "action"
                ],
                "claimed_slice_convention": None,
                "detected_order": None,
                "canonical_order": "next_state_previous_state_action",
                "contradiction": False,
                "reason": None,
                "source_order": "next_state_previous_state"
            },
            "D": {
                "source": "InitialParameterization",
                "shape": [
                    2
                ],
                "derived": False
            },
            "C": {
                "source": "passive_model_adapter",
                "shape": [
                    2
                ],
                "derived": True,
                "reason": "zero preferences for passive HMM/Markov model"
            }
        },
        "state_factors": [
            {
                "name": "s_t",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Hidden state belief at time t",
                "index": 0,
                "role": "factor"
            },
            {
                "name": "s_prime",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Hidden state belief at time t+1",
                "index": 1,
                "role": "bookkeeping"
            }
        ],
        "observation_modalities": [
            {
                "name": "o_t",
                "size": 2,
                "dimensions": [
                    2,
                    1
                ],
                "type": "float",
                "comment": "Observation at time t",
                "index": 0,
                "role": "factor"
            }
        ],
        "control_factors": [],
        "adapter_notes": [
            "passive_model_zero_preferences"
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
                1
            ],
            "derived": False,
            "declared_order": [
                "next_state",
                "previous_state",
                "action"
            ],
            "claimed_slice_convention": None,
            "detected_order": None,
            "canonical_order": "next_state_previous_state_action",
            "contradiction": False,
            "reason": None,
            "source_order": "next_state_previous_state"
        },
        "D": {
            "source": "InitialParameterization",
            "shape": [
                2
            ],
            "derived": False
        },
        "C": {
            "source": "passive_model_adapter",
            "shape": [
                2
            ],
            "derived": True,
            "reason": "zero preferences for passive HMM/Markov model"
        }
    },
    "canonical_pomdp_schema": "canonical_pomdp_v1",
    "variables": [
        {
            "name": "s_t",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Hidden state belief at time t"
        },
        {
            "name": "s_prime",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Hidden state belief at time t+1"
        },
        {
            "name": "t",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Discrete time index"
        },
        {
            "name": "o_t",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Observation at time t"
        }
    ],
    "connections": [
        {
            "source": "D",
            "relation": ">",
            "target": "s_t"
        },
        {
            "source": "s_t",
            "relation": "-",
            "target": "A"
        },
        {
            "source": "A",
            "relation": "-",
            "target": "o_t"
        },
        {
            "source": "s_t",
            "relation": "-",
            "target": "B"
        },
        {
            "source": "B",
            "relation": ">",
            "target": "s_prime"
        },
        {
            "source": "s_t",
            "relation": "-",
            "target": "F"
        },
        {
            "source": "o_t",
            "relation": "-",
            "target": "F"
        }
    ],
    "ontology_mapping": {
        "A": "RecognitionMatrix",
        "B": "TransitionMatrix",
        "D": "Prior",
        "s_t": "HiddenState",
        "s_prime": "NextHiddenState",
        "o_t": "Observation",
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
    gnn_spec["model_parameters"].setdefault("num_timesteps", 10)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Dynamic Perception Model"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Dynamic Perception Model")
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
