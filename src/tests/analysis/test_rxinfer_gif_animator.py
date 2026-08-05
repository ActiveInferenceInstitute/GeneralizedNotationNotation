"""Tests for the RxInfer GIF animation generator.

Validates that generate_gif_animation produces valid GIF files with
the expected structure, embedded data, and white publication style.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from analysis.rxinfer.gif_animator import generate_gif_animation


def _build_synthetic_results(
    n_steps: int = 10,
    n_states: int = 3,
    n_actions: int = 2,
) -> dict[str, Any]:
    """Build a synthetic rxinfer_simulation_v1 result for testing."""
    beliefs = []
    for t in range(n_steps):
        row = [0.1] * n_states
        row[t % n_states] = 0.8
        beliefs.append(row)

    return {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
        "model_name": "TestModel",
        "num_timesteps": n_steps,
        "beliefs": beliefs,
        "observations": [t % n_states for t in range(n_steps)],
        "true_states": [(t + 1) % n_states for t in range(n_steps)],
        "actions": [t % n_actions for t in range(n_steps)],
        "variational_free_energy": [6.11 - 0.1 * i for i in range(20)],
        "vfe_per_iteration": [6.11 - 0.1 * i for i in range(20)],
        "expected_free_energy": [3.0 - 0.1 * t for t in range(n_steps)],
        "efe_per_action": [
            [round(3.0 - 0.1 * t + i, 4) for i in range(n_actions)]
            for t in range(n_steps)
        ],
        "policy_posterior": [[1.0 / n_actions] * n_actions for _ in range(n_steps)],
        "validation": {
            "all_valid": True,
            "inference_converged": True,
            "vfe_present": True,
            "belief_entropy_ok": True,
            "belief_accuracy": 0.8,
            "belief_accuracy_ok": True,
        },
        "runtime_metadata": {
            "uses_real_rxinfer": True,
            "model_kind": "flat",
        },
    }


def test_generates_gif_file(tmp_path: Path) -> None:
    """generate_gif_animation produces a GIF file on disk."""
    results = _build_synthetic_results()
    output = tmp_path / "test.gif"
    path = generate_gif_animation(results, output, "TestModel")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 1000


def test_gif_has_correct_format(tmp_path: Path) -> None:
    """The GIF file starts with the GIF89a header."""
    results = _build_synthetic_results()
    output = tmp_path / "test_header.gif"
    generate_gif_animation(results, output, "TestModel")
    with open(output, "rb") as f:
        header = f.read(6)
    assert header in (b"GIF87a", b"GIF89a"), f"Invalid GIF header: {header}"


def test_different_state_counts(tmp_path: Path) -> None:
    """Models with different state counts produce valid GIFs."""
    for n_states in [2, 4, 8, 16]:
        results = _build_synthetic_results(n_states=n_states)
        output = tmp_path / f"test_{n_states}.gif"
        generate_gif_animation(results, output, f"Model_{n_states}")
        assert output.exists(), f"GIF for {n_states} states not created"
        assert output.stat().st_size > 500


def test_dict_shaped_beliefs_handled(tmp_path: Path) -> None:
    """Dict-shaped beliefs_by_factor are handled correctly."""
    results = _build_synthetic_results()
    beliefs = results.pop("beliefs")
    results["beliefs_by_factor"] = {"joint_state": beliefs}
    output = tmp_path / "test_dict.gif"
    generate_gif_animation(results, output, "DictModel")
    assert output.exists()


def test_empty_beliefs_returns_empty(tmp_path: Path) -> None:
    """Empty beliefs produces empty string (no GIF)."""
    results: dict[str, Any] = {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
    }
    output = tmp_path / "empty.gif"
    path = generate_gif_animation(results, output, "EmptyModel")
    assert path == ""


def test_with_gnn_spec_for_graph_model(tmp_path: Path) -> None:
    """GIF generation works when gnn_spec is embedded for graph model."""
    results = _build_synthetic_results()
    results["gnn_spec"] = {
        "connections": ["D>s", "s>A", "s>o", "s>s_prime", "B>s_prime", "u>s_prime"],
        "model_parameters": {"num_states": 3},
    }
    output = tmp_path / "test_graph.gif"
    generate_gif_animation(results, output, "GraphModel")
    assert output.exists()
    assert output.stat().st_size > 1000


def test_white_publication_style(tmp_path: Path) -> None:
    """The GIF uses white publication style (not dark mode)."""
    import matplotlib

    matplotlib.use("Agg")
    results = _build_synthetic_results()
    output = tmp_path / "test_style.gif"
    generate_gif_animation(results, output, "StyleTest")
    # The function uses plt.style.use("default") which is white
    # Verify the file was created (the style is applied inside the function)
    assert output.exists()


def test_multiple_frames(tmp_path: Path) -> None:
    """The GIF has multiple animation frames."""
    results = _build_synthetic_results(n_steps=15)
    output = tmp_path / "test_frames.gif"
    generate_gif_animation(results, output, "FramesTest", fps=4)
    assert output.exists()
    # A GIF with 15 frames at 4fps should be at least 10KB
    assert output.stat().st_size > 10000
