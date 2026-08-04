#!/usr/bin/env python3
"""
Comprehensive tests for the RxInfer Step-16 analyzer visualizations.

Validates that ``create_rxinfer_visualizations`` robustly renders a
comprehensive, consistent PNG set for synthetic ``rxinfer_simulation_v1``
results, handles both dict-shaped array keys and missing optional keys
(``policy_posterior`` / ``expected_free_energy``) gracefully, and that every
returned file exists on disk with nonzero size.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.rxinfer.analyzer import create_rxinfer_visualizations
from analysis.viz_base import MATPLOTLIB_AVAILABLE

pytestmark = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib required")

# The full plot set a complete rxinfer result should emit.
EXPECTED_PLOT_TYPES = [
    "belief_evolution",
    "obs_vs_true",
    "belief_heatmap",
    "belief_entropy",
    "accuracy",
    "action_frequencies",
    "belief_convergence",
    "belief_trace",
    "free_energy",
    "observations",
]

# Core plots that must always appear when beliefs/observations/true_states exist,
# even if optional keys (policy_posterior, expected_free_energy) are missing.
CORE_PLOT_TYPES = [
    "belief_evolution",
    "obs_vs_true",
    "belief_heatmap",
    "belief_entropy",
]


def build_rxinfer_results(
    *,
    n_steps: int = 15,
    n_states: int = 3,
    n_actions: int = 4,
    include_optional: bool = True,
    dict_shaped: bool = False,
    include_core: bool = True,
) -> dict[str, Any]:
    """Build a realistic synthetic ``rxinfer_simulation_v1`` results dict.

    Beliefs are rows (n_steps x n_states) that shift from a uniform prior toward
    a peaked posterior so plots have visible structure. Observations, true
    states and actions are produced as flat traces.

    Args:
        n_steps: Number of simulated timesteps.
        n_states: Number of hidden states.
        n_actions: Number of control/action options.
        include_optional: Whether to include optional keys
            (``policy_posterior``, ``expected_free_energy``, ``efe_per_action``).
        dict_shaped: When True, wrap the per-step arrays in a single-factor dict
            (e.g. ``beliefs_by_factor={"joint_state": ...}``) to exercise the
            dict-shaped json layout used by some exemplars.
        include_core: When False, produce an essentially empty result.
    """
    beliefs = []
    for t in range(n_steps):
        row = [1.0 / n_states] * n_states
        # Stretch toward the t-th state cycling through states.
        peak = t % n_states
        row = [0.02] * n_states
        row[peak] = 1.0 - 0.02 * (n_states - 1)
        beliefs.append(row)

    true_states = [t % n_states for t in range(n_steps)]
    observations = [t % n_states for t in range(n_steps)]
    actions = [t % n_actions for t in range(n_steps)]
    efe = [round(3.0 - 0.1 * t, 4) for t in range(n_steps)]
    efe_per_action = [
        [round(3.0 - 0.1 * t + i, 4) for i in range(n_actions)] for t in range(n_steps)
    ]
    policy_posterior = [
        [1.0 if i == (t % n_actions) else 0.0 for i in range(n_actions)]
        for t in range(n_steps)
    ]

    def _maybe(container: Any, key: str, value: Any) -> None:
        if include_optional:
            container[key] = value

    if dict_shaped:
        data: dict[str, Any] = {
            "schema_version": "rxinfer_simulation_v1",
            "success": True,
            "framework": "RxInfer.jl",
            "num_timesteps": n_steps,
            "beliefs_by_factor": {"joint_state": beliefs},
            "observations_by_modality": {"joint_observation": observations},
            "hidden_states_by_factor": {"joint_state": true_states},
            "actions_by_control_factor": {"joint_action": actions},
            "metrics": {
                "all_valid": True,
            },
            "model_parameters": {
                "num_states": n_states,
                "num_actions": n_actions,
            },
        }
    else:
        data = {
            "schema_version": "rxinfer_simulation_v1",
            "success": True,
            "framework": "RxInfer.jl",
            "num_timesteps": n_steps,
            "beliefs": beliefs,
            "observations": observations,
            "true_states": true_states,
            "actions": actions,
            "metrics": {
                "all_valid": True,
            },
            "model_parameters": {
                "num_states": n_states,
                "num_actions": n_actions,
            },
        }

    _maybe(data, "expected_free_energy", efe)
    _maybe(data, "efe_per_action", efe_per_action)
    _maybe(data, "policy_posterior", policy_posterior)
    return data


def _basenames(paths: list[str]) -> set[str]:
    return {Path(p).name for p in paths}


def test_full_result_produces_rich_png_set(tmp_path: Path) -> None:
    """A complete result yields >=6 PNG files that exist on disk."""
    results = build_rxinfer_results()
    out = tmp_path / "viz"
    paths = create_rxinfer_visualizations(results, out, "simple_mdp")

    assert isinstance(paths, list)
    assert len(paths) >= 6

    for path in paths:
        p = Path(path)
        assert p.exists(), f"returned file missing: {path}"
        assert p.stat().st_size > 0, f"returned file empty: {path}"

    names = _basenames(paths)
    for plot_type in EXPECTED_PLOT_TYPES:
        expected = f"simple_mdp_rxinfer_{plot_type}.png"
        assert expected in names, f"expected plot missing: {expected}"


def test_full_result_produces_all_ten_plot_types(tmp_path: Path) -> None:
    """The full 10-plot gridworld-equivalent set is emitted for a rich result."""
    results = build_rxinfer_results()
    out = tmp_path / "viz"
    paths = create_rxinfer_visualizations(results, out, "rich")
    names = _basenames(paths)

    assert len(EXPECTED_PLOT_TYPES) == 10
    for plot_type in EXPECTED_PLOT_TYPES:
        assert f"rich_rxinfer_{plot_type}.png" in names


def test_missing_optional_keys_still_yields_core_plots(tmp_path: Path) -> None:
    """A result WITHOUT policy_posterior / expected_free_energy does not raise
    and still yields the core plots."""
    results = build_rxinfer_results(include_optional=False)
    out = tmp_path / "viz"
    # Must not raise.
    paths = create_rxinfer_visualizations(results, out, "sparse")

    assert isinstance(paths, list)
    assert len(paths) >= 4

    names = _basenames(paths)
    for plot_type in CORE_PLOT_TYPES:
        expected = f"sparse_rxinfer_{plot_type}.png"
        assert expected in names, f"core plot missing: {expected}"

    for path in paths:
        assert Path(path).exists() and Path(path).stat().st_size > 0


def test_dict_shaped_result_handled(tmp_path: Path) -> None:
    """Dict-shaped keys (beliefs_by_factor etc.) render the same plot set."""
    results = build_rxinfer_results(dict_shaped=True)
    out = tmp_path / "viz"
    paths = create_rxinfer_visualizations(results, out, "dictmod")

    names = _basenames(paths)
    for plot_type in CORE_PLOT_TYPES:
        expected = f"dictmod_rxinfer_{plot_type}.png"
        assert expected in names, (
            f"core plot missing for dict-shaped result: {expected}"
        )
    for plot_type in EXPECTED_PLOT_TYPES:
        assert f"dictmod_rxinfer_{plot_type}.png" in names


def test_empty_result_does_not_raise(tmp_path: Path) -> None:
    """An essentially empty result returns an empty (or minimal) list and does
    not raise."""
    results: dict[str, Any] = {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
    }
    out = tmp_path / "viz"
    paths = create_rxinfer_visualizations(results, out, "empty")
    assert isinstance(paths, list)


def test_synthetic_array_lengths_match(tmp_path: Path) -> None:
    """Sanity check the helper produces arrays of expected shapes."""
    results = build_rxinfer_results(n_steps=15, n_states=3, n_actions=4)
    assert len(results["beliefs"]) == 15
    assert all(len(row) == 3 for row in results["beliefs"])
    assert len(results["observations"]) == 15
    assert len(results["true_states"]) == 15
    assert len(results["actions"]) == 15
    assert len(results["expected_free_energy"]) == 15
    assert len(results["efe_per_action"]) == 15
    assert all(len(row) == 4 for row in results["efe_per_action"])
    assert len(results["policy_posterior"]) == 15
