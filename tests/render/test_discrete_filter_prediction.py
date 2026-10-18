"""Independent native filtering/EFE witnesses and invalid-result controls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _spec(asymmetric: bool) -> dict:
    if asymmetric:
        a = [[0.95, 0.25, 0.5], [0.05, 0.75, 0.5]]
        b = np.stack(
            [
                [[0.7, 0.1, 0.2], [0.2, 0.6, 0.3], [0.1, 0.3, 0.5]],
                [[0.1, 0.3, 0.5], [0.4, 0.6, 0.2], [0.5, 0.1, 0.3]],
            ],
            axis=2,
        ).tolist()
        d = [0.6, 0.3, 0.1]
        c = [0.0, 1.0]
    else:
        a = np.eye(2).tolist()
        b = [[[0.0], [1.0]], [[1.0], [0.0]]]
        d = [1.0, 0.0]
        c = [0.0, 0.0]
    return {
        "modelName": "filter_prediction_witness",
        "model_parameters": {
            "num_hidden_states": len(d),
            "num_obs": len(a),
            "num_actions": len(b[0][0]),
            "num_timesteps": 12,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {"A": a, "B": b, "C": c, "D": d},
    }


def _render(backend: str, spec: dict, path: Path) -> None:
    if backend == "pytorch":
        pytest.importorskip("torch")
        from gnn.render.pytorch import render_gnn_to_pytorch as render
    else:
        pytest.importorskip("numpyro")
        from gnn.render.numpyro import render_gnn_to_numpyro as render
    okay, message, _ = render(spec, path)
    assert okay, message


def _run(backend: str, script: Path, output: Path, *, x64: bool, probe: str = ""):
    env = dict(
        os.environ,
        **{backend.upper() + "_OUTPUT_DIR": str(output)},
        JAX_ENABLE_X64=str(x64).lower(),
    )
    code = (
        "import runpy, sys\n"
        + ("import torch; torch.manual_seed(821)\n" if backend == "pytorch" else "")
        + (probe or "runpy.run_path(sys.argv[1], run_name='__main__')\n")
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(script)],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.env_heavy
@pytest.mark.parametrize(
    "backend,x64", [("pytorch", False), ("numpyro", False), ("numpyro", True)]
)
@pytest.mark.parametrize(
    "asymmetric", [False, True], ids=["deterministic-flip", "asymmetric"]
)
def test_native_filter_and_every_action_efe_match_numpy(
    tmp_path: Path, backend: str, x64: bool, asymmetric: bool
) -> None:
    spec = _spec(asymmetric)
    script = tmp_path / "model.py"
    _render(backend, spec, script)
    output = tmp_path / "native"
    run = _run(backend, script, output, x64=x64)
    assert run.returncode == 0, run.stderr
    result = json.loads((output / "simulation_results.json").read_text())
    assert result["validation"]["all_valid"]
    params = spec["initialparameterization"]
    a, b, c, posterior = [
        np.asarray(params[k], dtype=float) for k in ["A", "B", "C", "D"]
    ]
    pref = np.exp(c - c.max())
    pref /= pref.sum()
    expected_beliefs, expected_efe = [], []
    for t, obs in enumerate(result["observations"]):
        prior = posterior if t == 0 else b[:, :, result["actions"][t - 1]] @ posterior
        joint = a[obs] * prior
        assert joint.sum() > 0
        posterior = joint / joint.sum()
        expected_beliefs.append(posterior.copy())
        scores = []
        for action in range(b.shape[2]):
            predicted_state = b[:, :, action] @ posterior
            predicted_obs = a @ predicted_state
            ambiguity = -np.dot(predicted_state, np.sum(a * np.log(a + 1e-16), axis=0))
            risk = np.sum(
                predicted_obs * (np.log(predicted_obs + 1e-16) - np.log(pref + 1e-16))
            )
            scores.append(ambiguity + risk)
        expected_efe.append(scores)
    tolerance = 2e-6 if backend == "numpyro" and not x64 else 1e-10
    np.testing.assert_allclose(
        result["beliefs"], expected_beliefs, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        result["efe_history"], expected_efe, rtol=tolerance, atol=tolerance
    )
    assert len(result["beliefs"]) == 12
    if not asymmetric:
        np.testing.assert_allclose(
            result["beliefs"], np.tile(np.eye(2), (6, 1)), atol=tolerance
        )


@pytest.mark.env_heavy
@pytest.mark.parametrize("backend", ["pytorch", "numpyro"])
def test_zero_predictive_evidence_fails_before_fabricating_beliefs(
    tmp_path: Path, backend: str
) -> None:
    script = tmp_path / "model.py"
    _render(backend, _spec(False), script)
    probe = "module = runpy.run_path(sys.argv[1], run_name='probe')\nchoices = iter([0, 1])\n"
    if backend == "pytorch":
        probe += "module['torch'].multinomial = lambda *a, **k: module['torch'].tensor([next(choices)])\n"
    else:
        probe += "sampler = type(module['dist'].Categorical(probs=module['jnp'].asarray([1., 0.])))\n"
        probe += "sampler.sample = lambda self, *a, **k: module['jnp'].asarray(next(choices))\n"
    probe += "module['run_simulation']()\n"
    output = tmp_path / "native"
    run = _run(backend, script, output, x64=False, probe=probe)
    assert run.returncode != 0
    assert "zero or nonfinite predictive probability" in run.stderr
    assert not (output / "simulation_results.json").exists()


@pytest.mark.env_heavy
@pytest.mark.parametrize("backend", ["pytorch", "numpyro"])
def test_invalid_persisted_validation_cannot_exit_successfully(
    tmp_path: Path, backend: str
) -> None:
    script = tmp_path / "model.py"
    _render(backend, _spec(False), script)
    code = script.read_text()
    original = 'validation["all_valid"] = all(validation.values())'
    assert code.count(original) == 1
    # A false-result control exercises the public process boundary after saving
    # the artifact, independent of which numerical invariant rejected the run.
    script.write_text(code.replace(original, 'validation["all_valid"] = False'))
    output = tmp_path / "native"
    run = _run(backend, script, output, x64=False)
    assert run.returncode != 0
    assert "Invalid simulation results" in run.stderr
    assert (
        json.loads((output / "simulation_results.json").read_text())["validation"][
            "all_valid"
        ]
        is False
    )
