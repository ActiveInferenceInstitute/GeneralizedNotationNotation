"""Authored CPTs, native conditional queries, and refusal of silent demos."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gnn.execute.subprocess_envelope import run_subprocess_envelope
from gnn.render.bnlearn import generate_bnlearn_code
from gnn.render.bnlearn.source_program import source_payload
from gnn.render.processor import render_gnn_spec


def model():
    return {
        "model_name": "Three readings / two states / two controls",
        "model_parameters": {
            "num_timesteps": 4,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.8, 0.1], [0.15, 0.3], [0.05, 0.6]],
            "B": [[[0.9, 0.4], [0.2, 0.7]], [[0.1, 0.6], [0.8, 0.3]]],
            "C": [-1, 2, 0],
            "D": [0.7, 0.3],
        },
    }


def saved_payload(path):
    assignment = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.Assign) and n.targets[0].id == "PAYLOAD"
    )
    return json.loads(ast.literal_eval(assignment.value.args[0]))


def test_public_generator_preserves_authored_cpts_and_name(tmp_path):
    source = model()
    original = deepcopy(source)
    ok, reason, files = render_gnn_spec(source, "bnlearn", tmp_path)
    assert ok, reason
    payload = saved_payload(Path(files[0]))
    assert payload["model_name"] == source["model_name"]
    assert payload["source_parameters"] == source["initialparameterization"]
    assert payload["num_timesteps"] == 4
    assert source == original
    assert "parameter_learning.fit" not in Path(files[0]).read_text()
    changed = deepcopy(source)
    changed["initialparameterization"]["A"] = [[0.7, 0.2], [0.2, 0.3], [0.1, 0.5]]
    assert (
        source_payload(changed)["model_contract_sha256"]
        != payload["model_contract_sha256"]
    )


def test_passive_model_without_preferences_is_a_conditional_filter():
    source = model()
    initial = source["initialparameterization"]
    initial["B"] = np.asarray(initial["B"])[:, :, 0].tolist()
    del initial["C"]
    payload = source_payload(source)
    assert payload["matrix_provenance"]["C"]["derived"] is True
    assert "C" not in payload["source_parameters"]
    assert np.asarray(payload["B"]).shape == (2, 2, 1)


@pytest.mark.parametrize(
    "failure", ["missing", "mass", "orientation", "boolean", "component", "workload"]
)
def test_invalid_scientific_input_never_falls_back_or_replaces_a_program(
    tmp_path, failure
):
    source = model()
    if failure == "missing":
        del source["initialparameterization"]["D"]
    elif failure == "mass":
        source["initialparameterization"]["A"][0][0] = 4
    elif failure == "orientation":
        del source["model_parameters"]["b_tensor_order"]
    elif failure == "boolean":
        source["model_parameters"]["num_timesteps"] = True
    elif failure == "workload":
        source["model_parameters"]["num_timesteps"] = 100_001
    else:
        source["initialparameterization"]["A_agent1"] = [[1]]
    path = tmp_path / "previous.py"
    path.write_text("previous program")
    with pytest.raises(ValueError):
        generate_bnlearn_code(source, path)
    assert path.read_text() == "previous program"


def test_metadata_only_compatibility_demo_labels_its_scientific_limit():
    code = generate_bnlearn_code({"model_name": "backend smoke"})
    compile(code, "demo.py", "exec")
    assert '"source_parameter_fidelity": False' in code
    assert "synthetic_backend_smoke_v1" in code


@pytest.mark.needs_bnlearn
@pytest.mark.parametrize("passive", [False, True])
def test_native_bnlearn_posteriors_match_independent_forward_filter(tmp_path, passive):
    source = model()
    if passive:
        raw = source["initialparameterization"]
        raw["B"] = np.asarray(raw["B"])[:, :, 0].tolist()
        del raw["C"]
    readings = [2, 0, 1, 2]
    actions = [0, 0, 0] if passive else [1, 0, 1]
    script = tmp_path / "authored.py"
    generate_bnlearn_code(source, script)
    (tmp_path / "observations.json").write_text(json.dumps(readings))
    (tmp_path / "actions.json").write_text(json.dumps(actions))
    env = dict(os.environ, BNLEARN_OUTPUT_DIR=str(tmp_path / "results"))
    result = run_subprocess_envelope(
        [
            sys.executable,
            str(script),
            "--observations-json",
            str(tmp_path / "observations.json"),
            "--transition-actions-json",
            str(tmp_path / "actions.json"),
        ],
        cwd=tmp_path,
        env=env,
        timeout=60,
        sandbox=False,
    )
    assert result["success"], result
    saved = json.loads((tmp_path / "results/simulation_results.json").read_text())
    initial = source["initialparameterization"]
    a, b, prior = (np.asarray(initial[k], dtype=float) for k in ("A", "B", "D"))
    if b.ndim == 2:
        b = b[:, :, None]
    expected = []
    for step, reading in enumerate(readings):
        predicted = prior if step == 0 else b[:, :, actions[step - 1]] @ prior
        posterior = a[reading] * predicted
        prior = posterior / posterior.sum()
        expected.append(prior)
    np.testing.assert_allclose(saved["beliefs"], expected, rtol=1e-12, atol=1e-14)
    assert saved["true_states"] == []
    assert saved["observations"] == readings
    assert saved["transition_actions"] == actions
    assert saved["inference_contract"] == "bnlearn_fixed_action_filter_v1"
    assert saved["dtype"] == "float64"
    assert saved["initial_time"] == "first_observation_uses_prior"
    assert saved["conditioning_horizon"] == [1, 2, 3, 4]
    assert saved["axes"] == {
        "A": ["observation", "state"],
        "B": ["next_state", "previous_state", "action"],
    }
    assert saved["belief_axes"] == ["timestep", "state"]
    assert saved["parameters"] == {
        "A": a.tolist(),
        "B": b.tolist(),
        "D": np.asarray(initial["D"]).tolist(),
    }
    assert (
        saved["generated_script_sha256"]
        == hashlib.sha256(script.read_bytes()).hexdigest()
    )
    assert saved["python_version"] and saved["numpy_version"]
    assert "actions" not in saved
    from gnn.analysis.framework_extractors import extract_bnlearn_data

    extracted = extract_bnlearn_data(saved)
    assert extracted["beliefs"] == saved["beliefs"]
    assert extracted["transition_actions"] == actions
    assert saved["num_timesteps"] == source["model_parameters"]["num_timesteps"]
    assert saved["source_parameters"] == initial
    assert saved["validation"]["all_valid"] is True
    assert saved["source_parameter_fidelity"] is True


def test_public_file_summary_rehydrates_authored_parameters(tmp_path):
    from gnn import parse_gnn_file

    path = (
        Path(__file__).resolve().parents[2]
        / "input/gnn_files/discrete/bnlearn_causal_model.md"
    )
    summary = parse_gnn_file(path)
    ok, message, artifacts = render_gnn_spec(summary, "bnlearn", tmp_path)
    assert ok, message
    payload = saved_payload(Path(artifacts[0]))
    assert payload["inference_contract"] == "bnlearn_fixed_action_filter_v1"
    assert payload["model_name"] == "Bnlearn Causal Model"
    assert payload["num_timesteps"] == 30
    assert np.asarray(payload["A"]).shape == (2, 2)


@pytest.mark.needs_bnlearn
def test_impossible_observation_fails_without_writing_a_success_receipt(tmp_path):
    source = model()
    source["model_parameters"]["num_timesteps"] = 1
    source["initialparameterization"]["A"] = [[1, 1], [0, 0]]
    source["initialparameterization"]["C"] = [0, 0]
    script = tmp_path / "impossible.py"
    generate_bnlearn_code(source, script)
    observations = tmp_path / "observations.json"
    observations.write_text("[1]")
    output = tmp_path / "results"
    result = run_subprocess_envelope(
        [sys.executable, str(script), "--observations-json", str(observations)],
        cwd=tmp_path,
        env=dict(os.environ, BNLEARN_OUTPUT_DIR=str(output)),
        timeout=60,
        sandbox=False,
    )
    assert not result["success"]
    assert "invalid posterior" in result["stderr"]
    assert not (output / "simulation_results.json").exists()


@pytest.mark.parametrize(
    "matrices",
    [
        {"A_f0": [[1]], "B_f0": [[1]], "D_f0": [1]},
        {"F": [[1]], "H": [[1]], "Q": [[1]], "R": [[1]]},
    ],
)
def test_scientific_components_never_emit_metadata_smoke(matrices):
    with pytest.raises(ValueError, match="unsupported-bnlearn"):
        generate_bnlearn_code({"initialparameterization": matrices})
    with pytest.raises(ValueError, match="unsupported-bnlearn"):
        generate_bnlearn_code({"structured_pomdp": {"matrices": matrices}})
