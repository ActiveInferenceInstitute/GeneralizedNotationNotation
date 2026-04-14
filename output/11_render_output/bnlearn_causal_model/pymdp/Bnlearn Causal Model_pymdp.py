#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for Bnlearn Causal Model

This file was generated from a GNN specification by
``src/render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``src.execute.pymdp.run_simple_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        Bnlearn Causal Model
Description:  
Generated:    2026-04-14 11:53:47

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
            "ERROR: Cannot locate GNN repository root. Run via the pipeline "
            "execute step, or set GNN_PROJECT_ROOT to the checkout root.",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.path.insert(0, str(_found))

# ---------------------------------------------------------------------------
# pymdp 1.0.0 presence check (hard requirement)
# ---------------------------------------------------------------------------
try:
    import pymdp  # noqa: F401
    from pymdp.agent import Agent  # noqa: F401
    if not hasattr(Agent, "update_empirical_prior"):
        raise ImportError("legacy pymdp (<1.0.0) detected")
    print("PyMDP 1.0.0+ detected (JAX-first Agent).")
except ImportError as e:
    print(
        "ERROR: pymdp 1.0.0 required. Install with: "
        "uv pip install 'inferactively-pymdp>=1.0.0' (original error: "
        + str(e) + ")",
        file=sys.stderr,
    )
    sys.exit(1)

from src.execute.pymdp import execute_pymdp_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run a pymdp 1.0.0 simulation for the GNN model embedded in this file."""
    # Matrices embedded verbatim from the GNN spec.
    A_data = [[0.9, 0.1], [0.1, 0.9]]
    B_data = [[[0.7, 0.3], [0.3, 0.7]], [[0.3, 0.7], [0.7, 0.3]]]
    C_data = [0.0, 1.0]
    D_data = [0.5, 0.5]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "Bnlearn Causal Model",
    "model_name": "Bnlearn Causal Model",
    "description": "A Bayesian Network model mapping Active Inference structure:\n- S: Hidden State\n- A: Action\n- S_prev: Previous State\n- O: Observation",
    "model_parameters": {
        "num_hidden_states": 2,
        "num_obs": 2,
        "num_actions": 2,
        "simulation_params": {},
        "num_timesteps": 15
    },
    "initialparameterization": {
        "A": [
            [
                0.9,
                0.1
            ],
            [
                0.1,
                0.9
            ]
        ],
        "B": [
            [
                [
                    0.7,
                    0.3
                ],
                [
                    0.3,
                    0.7
                ]
            ],
            [
                [
                    0.3,
                    0.7
                ],
                [
                    0.7,
                    0.3
                ]
            ]
        ],
        "C": [
            0.0,
            1.0
        ],
        "D": [
            0.5,
            0.5
        ]
    },
    "variables": [
        {
            "name": "s",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": None
        },
        {
            "name": "s_prev",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": None
        },
        {
            "name": "a",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": None
        },
        {
            "name": "o",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": None
        }
    ],
    "connections": [
        {
            "source": "s_prev",
            "relation": ">",
            "target": "s"
        },
        {
            "source": "a",
            "relation": ">",
            "target": "s"
        },
        {
            "source": "s",
            "relation": ">",
            "target": "o"
        }
    ],
    "ontology_mapping": {
        "A": "ObservationModel",
        "B": "TransitionModel",
        "s": "HiddenState",
        "s_prev": "PreviousState",
        "o": "Observation",
        "a": "Action"
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

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/Bnlearn Causal Model"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for Bnlearn Causal Model")
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
