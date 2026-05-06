"""Focused tests for the PyTorch render backend."""

from __future__ import annotations

import py_compile


def _small_gnn_spec() -> dict:
    return {
        "modelName": "pytorch_smoke",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_timesteps": 7,
        },
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [
                [[0.9, 0.2], [0.1, 0.8]],
                [[0.8, 0.1], [0.2, 0.9]],
            ],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
    }


def test_pytorch_renderer_generates_compilable_script_with_timestep(tmp_path) -> None:
    from render.pytorch.pytorch_renderer import render_gnn_to_pytorch

    output_path = tmp_path / "pytorch_smoke.py"

    success, message, artifacts = render_gnn_to_pytorch(_small_gnn_spec(), output_path)

    assert success, message
    assert artifacts == [str(output_path)]
    generated = output_path.read_text(encoding="utf-8")
    assert "T = 7" in generated
    py_compile.compile(str(output_path), doraise=True)


def test_render_success_policy_can_be_strict_about_framework_failures() -> None:
    from render.processor import _render_succeeded

    assert _render_succeeded(
        success_count=1,
        total_files=1,
        total_framework_successes=1,
        total_framework_attempts=2,
        strict_framework_success=False,
    ) is True
    assert _render_succeeded(
        success_count=1,
        total_files=1,
        total_framework_successes=1,
        total_framework_attempts=2,
        strict_framework_success=True,
    ) is False
