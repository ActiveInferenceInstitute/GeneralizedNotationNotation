#!/usr/bin/env python3
"""
pymdp 1.0.0 runner for T-Maze Epistemic Foraging Agent

This file was generated from a GNN specification by
``src/render/pymdp/pymdp_renderer.py``. It delegates the actual rollout
to the GNN pipeline's tested execution module
(``src.execute.pymdp.run_simple_pymdp_simulation``), which in turn calls
real pymdp 1.0.0 (JAX-first) under the hood.

Model:        T-Maze Epistemic Foraging Agent
Description:  
Generated:    2026-04-15 12:26:34

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
    A_data = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    B_data = [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]
    C_data = [0.0, 0.0, 0.0, 0.0]
    D_data = [1.0, 0.0, 0.0, 0.0]
    E_data = None

    # Full parsed spec, with matrices merged into initialparameterization.
    gnn_spec = {
    "name": "T-Maze Epistemic Foraging Agent",
    "model_name": "T-Maze Epistemic Foraging Agent",
    "description": "The classic T-maze task from Active Inference literature (Friston et al.):\n- Agent navigates a T-shaped maze with 4 locations: center, left arm, right arm, cue location\n- Two observation modalities: location (where am I?) and reward/cue (what do I see?)\n- Reward is hidden behind one of the two arms (left or right), determined by context\n- Cue location provides partial information about which arm holds the reward\n- Agent must decide: go directly to an arm (exploit) or visit cue location first (explore)\n- Demonstrates epistemic foraging: Active Inference naturally balances exploration vs exploitation\n- The Expected Free Energy decomposes into epistemic (information gain) + instrumental (reward) value",
    "model_parameters": {
        "num_hidden_states": 4,
        "num_obs": 4,
        "num_actions": 4,
        "simulation_params": {},
        "num_timesteps": 15
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
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    1.0,
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
                    1.0,
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
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
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
                    1.0,
                    1.0,
                    0.0
                ]
            ]
        ],
        "C": [
            0.0,
            0.0,
            0.0,
            0.0
        ],
        "D": [
            1.0,
            0.0,
            0.0,
            0.0
        ]
    },
    "variables": [
        {
            "name": "s_loc",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Location state: (0:center, 1:left_arm, 2:right_arm, 3:cue_location)"
        },
        {
            "name": "s_ctx",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Context state: (0:reward_left, 1:reward_right)"
        },
        {
            "name": "A_loc",
            "dimensions": [
                4,
                4
            ],
            "type": "float",
            "comment": "Location likelihood: P(o_loc | s_loc) \u2014 identity"
        },
        {
            "name": "A_rew",
            "dimensions": [
                3,
                4,
                2
            ],
            "type": "float",
            "comment": "Reward likelihood: P(o_rew | s_loc, s_ctx) \u2014 context-dependent"
        },
        {
            "name": "B_ctx",
            "dimensions": [
                2,
                2,
                1
            ],
            "type": "float",
            "comment": "Context transitions: identity (context doesn't change)"
        },
        {
            "name": "C_loc",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "No location preference (agent doesn't prefer a location per se)"
        },
        {
            "name": "D_loc",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Prior: agent starts at center"
        },
        {
            "name": "D_ctx",
            "dimensions": [
                2
            ],
            "type": "float",
            "comment": "Prior: uncertain about which arm has reward"
        },
        {
            "name": "G_epi",
            "dimensions": [
                "pi"
            ],
            "type": "float",
            "comment": "Epistemic value (information gain about context)"
        },
        {
            "name": "G_ins",
            "dimensions": [
                "pi"
            ],
            "type": "float",
            "comment": "Instrumental value (expected reward)"
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
            "name": "o_loc",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Location observation: (0:center, 1:left, 2:right, 3:cue)"
        },
        {
            "name": "o_rew",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Reward/cue observation: (0:no_reward, 1:reward, 2:cue_left)"
        },
        {
            "name": "C_rew",
            "dimensions": [
                3
            ],
            "type": "float",
            "comment": "Reward preference: strongly prefers reward observation"
        },
        {
            "name": "B_loc",
            "dimensions": [
                4,
                4,
                4
            ],
            "type": "float",
            "comment": "Location transitions: P(s_loc' | s_loc, action)"
        },
        {
            "name": "pi",
            "dimensions": [
                4
            ],
            "type": "float",
            "comment": "Policy over 4 actions: (go_left, go_right, go_cue, stay)"
        },
        {
            "name": "u",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Selected action"
        },
        {
            "name": "G",
            "dimensions": [
                "pi"
            ],
            "type": "float",
            "comment": "Expected Free Energy per policy"
        }
    ],
    "connections": [
        {
            "source": "D_loc",
            "relation": ">",
            "target": "s_loc"
        },
        {
            "source": "D_ctx",
            "relation": ">",
            "target": "s_ctx"
        },
        {
            "source": "s_loc",
            "relation": "-",
            "target": "A_loc"
        },
        {
            "source": "A_loc",
            "relation": "-",
            "target": "o_loc"
        },
        {
            "source": "s_loc",
            "relation": "-",
            "target": "A_rew"
        },
        {
            "source": "s_ctx",
            "relation": "-",
            "target": "A_rew"
        },
        {
            "source": "A_rew",
            "relation": "-",
            "target": "o_rew"
        },
        {
            "source": "s_loc",
            "relation": "-",
            "target": "B_loc"
        },
        {
            "source": "s_ctx",
            "relation": "-",
            "target": "B_ctx"
        },
        {
            "source": "C_rew",
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
            "source": "G",
            "relation": ">",
            "target": "pi"
        },
        {
            "source": "pi",
            "relation": ">",
            "target": "u"
        },
        {
            "source": "B_loc",
            "relation": ">",
            "target": "u"
        },
        {
            "source": "s_loc",
            "relation": "-",
            "target": "F"
        },
        {
            "source": "s_ctx",
            "relation": "-",
            "target": "F"
        },
        {
            "source": "o_loc",
            "relation": "-",
            "target": "F"
        },
        {
            "source": "o_rew",
            "relation": "-",
            "target": "F"
        }
    ],
    "ontology_mapping": {
        "A_loc": "LocationLikelihoodMatrix",
        "A_rew": "RewardLikelihoodMatrix",
        "B_loc": "LocationTransitionMatrix",
        "B_ctx": "ContextTransitionMatrix",
        "C_loc": "LocationPreferenceVector",
        "C_rew": "RewardPreferenceVector",
        "D_loc": "LocationPrior",
        "D_ctx": "ContextPrior",
        "s_loc": "LocationHiddenState",
        "s_ctx": "ContextHiddenState",
        "o_loc": "LocationObservation",
        "o_rew": "RewardObservation",
        "pi": "PolicyVector",
        "u": "Action",
        "G": "ExpectedFreeEnergy",
        "G_epi": "EpistemicValue",
        "G_ins": "InstrumentalValue",
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
    gnn_spec["model_parameters"].setdefault("num_timesteps", 15)

    output_dir = Path(os.environ.get("PYMDP_OUTPUT_DIR", "output/pymdp_simulations/T-Maze Epistemic Foraging Agent"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running pymdp 1.0.0 rollout for T-Maze Epistemic Foraging Agent")
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
