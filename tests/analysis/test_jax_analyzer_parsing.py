"""Pins for JAX analyzer parsing (``parse_raw_output``, results loading).

Previously at 7% coverage: the regex contract over raw JAX execution stdout
and the three-path ``simulation_results.json`` lookup were untested.
"""

from __future__ import annotations

import json
from pathlib import Path

from gnn.analysis.jax.analyzer import load_simulation_results_json, parse_raw_output

RAW_OUTPUT = """\
JAX Active Inference Simulation
Actions taken: [0 1 0 1 0 0]
Final belief: [0.85 0.15]
Average EFE: 0.000312
EFE for all actions: [0.00986 0.00962]
A matrix shape: (2, 3)
B matrix shape: (2, 2, 2)
C vector shape: (3,)
D vector shape: (2,)
Number of states: 2
Number of observations: 3
Number of actions: 2
"""


def test_parse_raw_output_extracts_all_documented_fields() -> None:
    extracted = parse_raw_output(RAW_OUTPUT)

    assert extracted["actions_from_output"] == [0, 1, 0, 1, 0, 0]
    assert extracted["num_simulation_steps"] == 6
    assert extracted["final_belief"] == [0.85, 0.15]
    assert extracted["average_efe"] == 0.000312
    assert extracted["efe_all_actions"] == [0.00986, 0.00962]
    assert extracted["model_shapes"]["A_shape"] == (2, 3)
    assert extracted["model_shapes"]["B_shape"] == (2, 2, 2)
    assert extracted["model_shapes"]["C_shape"] == (3,)
    assert extracted["model_shapes"]["D_shape"] == (2,)
    assert extracted["model_shapes"]["num_states"] == 2
    assert extracted["model_shapes"]["num_observations"] == 3
    assert extracted["model_shapes"]["num_actions"] == 2


def test_parse_raw_output_empty_returns_zeroed_structure() -> None:
    extracted = parse_raw_output("")

    assert extracted == {
        "actions_from_output": [],
        "final_belief": [],
        "average_efe": None,
        "efe_all_actions": [],
        "model_shapes": {},
        "num_simulation_steps": 0,
    }


def test_parse_raw_output_unparseable_text_is_tolerated() -> None:
    extracted = parse_raw_output("complete garbage without any markers")

    assert extracted["actions_from_output"] == []
    assert extracted["average_efe"] is None
    assert extracted["model_shapes"] == {}


def test_load_simulation_results_json_finds_each_location(tmp_path: Path) -> None:
    payload = {"schema_version": "jax_simulation_v1", "beliefs": [0.5]}

    for location in (
        tmp_path / "simulation_results.json",
        tmp_path / "simulation_data" / "simulation_results.json",
        tmp_path / "jax_outputs" / "simulation_results.json",
    ):
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(json.dumps(payload), encoding="utf-8")
        assert load_simulation_results_json(tmp_path) == payload

    assert load_simulation_results_json(tmp_path / "nowhere") is None


def test_load_simulation_results_json_tolerates_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "simulation_results.json").write_text("{broken", encoding="utf-8")

    assert load_simulation_results_json(tmp_path) is None
