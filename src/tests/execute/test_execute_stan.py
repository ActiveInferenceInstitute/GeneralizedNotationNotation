"""Stan executor: discovery, dependency gating, and a real CmdStan run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execute.stan import execute_stan_script, find_stan_scripts, is_stan_available
from render.stan.stan_renderer import render_gnn_to_stan

DISCRETE_SPEC = {
    "name": "Tiny HMM",
    "model_name": "Tiny HMM",
    "initialparameterization": {
        "A": [[0.85, 0.15], [0.15, 0.85]],
        "B": [[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]],
        "C": [0.0, 1.0],
        "D": [0.5, 0.5],
    },
    "model_parameters": {
        "num_hidden_states": 2,
        "num_obs": 2,
        "num_actions": 2,
        "num_timesteps": 6,
        "random_seed": 3,
    },
}


def test_find_stan_scripts_only_matches_drivers_in_stan_dirs(tmp_path: Path) -> None:
    ok, _, arts = render_gnn_to_stan(
        DISCRETE_SPEC, tmp_path / "m" / "stan" / "m_stan.py"
    )
    assert ok
    (tmp_path / "m" / "jax").mkdir()
    (tmp_path / "m" / "jax" / "other_stan.py").write_text("print()")
    found = find_stan_scripts(tmp_path)
    assert found == [Path(arts[0])]


def test_render_summary_contract_treats_stan_program_as_companion(tmp_path: Path) -> None:
    """The .stan program beside the driver is not a 'missing executable script'."""
    import json
    import logging

    from execute.processor import _load_render_summary_contract

    render_dir = tmp_path / "11"
    ok, _, arts = render_gnn_to_stan(
        DISCRETE_SPEC, render_dir / "tiny" / "stan" / "tiny_stan.py"
    )
    assert ok
    summary = {
        "file_results": {
            "input/tiny.md": {
                "framework_results": {
                    "stan": {"success": True, "output_files": arts},
                    "pymdp": {
                        "success": False,
                        "unsupported": True,
                        "message": "continuous-state model",
                    },
                }
            }
        }
    }
    (render_dir / "render_processing_summary.json").write_text(json.dumps(summary))
    allowed, failures = _load_render_summary_contract(
        render_dir, ["stan", "pymdp"], logging.getLogger("t"), target_dir=None
    )
    assert allowed == {Path(arts[0]).resolve()}
    assert failures == []


def test_discrete_program_declares_forward_algorithm(tmp_path: Path) -> None:
    ok, _, arts = render_gnn_to_stan(DISCRETE_SPEC, tmp_path / "m_stan.py")
    assert ok
    text = Path(arts[1]).read_text()
    assert "dirichlet(alpha_A[s])" in text
    assert "filtered_state" in text and "log_marginal" in text
    assert "B[u[t - 1]] * alpha" in text  # vectorised scaled forward recursion


def test_stan_execution_end_to_end(tmp_path: Path) -> None:
    if not is_stan_available():
        pytest.skip("cmdstanpy/CmdStan not installed")
    ok, _, arts = render_gnn_to_stan(DISCRETE_SPEC, tmp_path / "m_stan.py")
    assert ok
    result = execute_stan_script(arts[0], tmp_path / "out", timeout=900)
    assert result["success"], result["stderr"][-2000:]
    res = json.loads(Path(result["results_file"]).read_text())
    assert res["framework"] == "stan" and res["model_kind"] == "discrete"
    assert len(res["beliefs"]) == 6 and len(res["beliefs"][0]) == 2
    assert len(res["A_posterior_mean"]) == 2 and len(res["A_posterior_mean"][0]) == 2
    assert res["validation"]["all_valid"] is True
