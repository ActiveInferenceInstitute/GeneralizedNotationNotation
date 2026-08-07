#!/usr/bin/env python3
"""Comprehensive tests for the RxInfer Step-16 analyzer visualizations.

Validates that ``create_rxinfer_visualizations`` robustly renders a
comprehensive, consistent PNG set for synthetic ``rxinfer_simulation_v1``
results, handles both dict-shaped array keys and missing optional keys
(``policy_posterior`` / ``expected_free_energy``) gracefully, and that every
returned file exists on disk with nonzero size.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.rxinfer.analyzer import (
    compute_per_factor_beliefs,
    create_rxinfer_visualizations,
    generate_analysis_from_logs,
)

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
    "efe_per_action_heatmap",
]

# Core plots that must always appear when beliefs/observations/true_states exist,
# even if optional keys (policy_posterior, expected_free_energy, efe_per_action) are missing.
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

    assert len(EXPECTED_PLOT_TYPES) == 11
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


def build_factored_results(
    *,
    factor_sizes: tuple[tuple[str, int], ...] = (
        ("s_agent1", 4),
        ("s_agent2", 4),
        ("s_joint", 16),
    ),
    n_steps: int = 6,
) -> dict[str, Any]:
    """Build a multi-factor result whose beliefs live on the flattened joint space.

    The joint state count is the product of the factor sizes, matching the
    C-order ``itertools.product`` flattening the renderer uses. Beliefs are
    deterministic, strictly positive and normalised per timestep.
    """
    sizes = [size for _, size in factor_sizes]
    joint_size = math.prod(sizes)
    beliefs = []
    for t in range(n_steps):
        raw = [float(((t + 1) * (k + 3)) % 7) + 0.5 for k in range(joint_size)]
        total = sum(raw)
        beliefs.append([value / total for value in raw])

    return {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
        "num_timesteps": n_steps,
        "beliefs": beliefs,
        "observations": [t % 3 for t in range(n_steps)],
        "true_states": [t % 3 for t in range(n_steps)],
        "actions": [t % 2 for t in range(n_steps)],
        "model_parameters": {
            "num_states": joint_size,
            "num_state_factors": len(factor_sizes),
            "state_factors": [
                {"name": name, "size": size} for name, size in factor_sizes
            ],
        },
    }


def _joint_from_marginals(marginals: list[list[float]]) -> list[float]:
    """Flatten an outer product of marginals in C order (first factor slowest)."""
    joint = [1.0]
    for marginal in marginals:
        joint = [value * component for value in joint for component in marginal]
    return joint


def test_per_factor_marginals_sum_to_one() -> None:
    """Every factor marginal is a normalised distribution at every timestep."""
    results = build_factored_results(n_steps=5)
    per_factor = compute_per_factor_beliefs(results)

    assert set(per_factor) == {"s_agent1", "s_agent2", "s_joint"}
    expected_sizes = {"s_agent1": 4, "s_agent2": 4, "s_joint": 16}
    for name, trajectory in per_factor.items():
        assert len(trajectory) == 5, f"{name} lost timesteps"
        for step, marginal in enumerate(trajectory):
            assert len(marginal) == expected_sizes[name]
            assert abs(sum(marginal) - 1.0) < 1e-9, (
                f"{name} marginal at t={step} sums to {sum(marginal)}"
            )
            assert all(value >= 0.0 for value in marginal)


def test_per_factor_recovers_hand_computed_c_order_marginals() -> None:
    """A 2x3 joint built by C-order outer product decomposes back to its marginals."""
    m_first = [0.3, 0.7]
    m_second = [0.2, 0.5, 0.3]
    joint = _joint_from_marginals([m_first, m_second])

    # C order: index = i * 3 + j, first factor slowest-varying.
    assert joint == pytest.approx(
        [0.3 * 0.2, 0.3 * 0.5, 0.3 * 0.3, 0.7 * 0.2, 0.7 * 0.5, 0.7 * 0.3]
    )

    results = {
        "beliefs": [joint, joint],
        "model_parameters": {
            "state_factors": [{"name": "s_a", "size": 2}, {"name": "s_b", "size": 3}],
        },
    }
    per_factor = compute_per_factor_beliefs(results)

    assert per_factor["s_a"][0] == pytest.approx(m_first)
    assert per_factor["s_b"][0] == pytest.approx(m_second)
    assert per_factor["s_a"][1] == pytest.approx(m_first)


def test_per_factor_returns_empty_without_state_factors() -> None:
    """Artifacts without a state_factors key are structurally flat: {}."""
    results = build_rxinfer_results()
    assert "state_factors" not in results["model_parameters"]
    assert compute_per_factor_beliefs(results) == {}

    del results["model_parameters"]
    assert compute_per_factor_beliefs(results) == {}


def test_per_factor_returns_empty_for_single_informative_factor() -> None:
    """One informative factor (with or without size-1 companions) yields {}."""
    single = build_factored_results(factor_sizes=(("s_only", 6),), n_steps=3)
    assert compute_per_factor_beliefs(single) == {}

    with_degenerate = build_factored_results(
        factor_sizes=(("s_only", 6), ("signal_decay", 1)), n_steps=3
    )
    assert compute_per_factor_beliefs(with_degenerate) == {}


def test_per_factor_raises_when_joint_composed_sizes_contradict() -> None:
    """A size mismatch in a JOINT-COMPOSED (multi_agent) payload is loud."""
    results = build_factored_results(
        factor_sizes=(("s_agent1", 4), ("s_agent2", 4)), n_steps=3
    )
    results["model_parameters"]["state_factors"][1]["size"] = 5
    results.setdefault("runtime_metadata", {})["model_kind"] = "multi_agent"

    with pytest.raises(ValueError, match="joint states"):
        compute_per_factor_beliefs(results)


def test_per_factor_descriptive_mismatch_is_structural_absence() -> None:
    """Non-composed kinds with descriptive factors (e.g. flat s/s_prime) get {}.

    Flat exemplars legitimately declare next-state aliases as two "factors"
    over a single chain; the belief space is NOT their product, so per-factor
    recovery does not apply — this must not raise (it broke GIF generation
    for most of the 46-model batch before the kind gate existed).
    """
    results = build_factored_results(factor_sizes=(("s", 3), ("s_prime", 3)), n_steps=3)
    # Beliefs are 3-wide (the chain), not 9-wide (the bogus product).
    results["beliefs"] = [[0.2, 0.3, 0.5] for _ in range(3)]
    results["beliefs_by_factor"] = {"joint_state": results["beliefs"]}
    results.setdefault("runtime_metadata", {})["model_kind"] = "flat"

    assert compute_per_factor_beliefs(results) == {}


def test_per_factor_raises_when_factor_descriptor_is_malformed() -> None:
    """A factor descriptor missing name/size is a spec error, not absence."""
    results = build_factored_results(
        factor_sizes=(("s_agent1", 4), ("s_agent2", 4)), n_steps=3
    )
    results["model_parameters"]["state_factors"][0] = {"size": 4}

    with pytest.raises(ValueError, match="name"):
        compute_per_factor_beliefs(results)


def test_size_one_factors_are_skipped_but_kept_in_the_reshape() -> None:
    """A size-1 factor contributes to flattening yet never reaches the output."""
    m_first = [0.5, 0.25, 0.25]
    m_second = [0.1, 0.6, 0.3]
    joint = _joint_from_marginals([m_first, m_second, [1.0]])
    assert len(joint) == 9

    results = {
        "beliefs": [joint],
        "model_parameters": {
            "state_factors": [
                {"name": "s_agent1", "size": 3},
                {"name": "s_agent2", "size": 3},
                {"name": "signal_decay", "size": 1},
            ],
        },
    }
    per_factor = compute_per_factor_beliefs(results)

    assert set(per_factor) == {"s_agent1", "s_agent2"}
    assert per_factor["s_agent1"][0] == pytest.approx(m_first)
    assert per_factor["s_agent2"][0] == pytest.approx(m_second)


def test_factored_result_emits_per_factor_plot_and_payload(tmp_path: Path) -> None:
    """The analyzer emits the per-factor PNG and stores the marginals on the payload."""
    results = build_factored_results(n_steps=5)
    out = tmp_path / "viz"
    paths = create_rxinfer_visualizations(results, out, "multi_agent")

    names = _basenames(paths)
    assert "multi_agent_rxinfer_per_factor_beliefs.png" in names
    plot = out / "multi_agent_rxinfer_per_factor_beliefs.png"
    assert plot.stat().st_size > 0

    payload = results["per_factor_beliefs"]
    assert set(payload) == {"s_agent1", "s_agent2", "s_joint"}
    assert len(payload["s_joint"]) == 5
    assert len(payload["s_joint"][0]) == 16


def test_flat_result_emits_no_per_factor_plot(tmp_path: Path) -> None:
    """Flat models get an empty per_factor_beliefs payload and no extra PNG."""
    results = build_rxinfer_results()
    out = tmp_path / "viz"
    paths = create_rxinfer_visualizations(results, out, "flat")

    assert results["per_factor_beliefs"] == {}
    assert "flat_rxinfer_per_factor_beliefs.png" not in _basenames(paths)


def test_generate_analysis_from_logs_emits_gif_and_per_factor(tmp_path: Path) -> None:
    """End-to-end: a results JSON in a log tree yields the GIF and per-factor outputs."""
    results = build_factored_results(
        factor_sizes=(("s_agent1", 4), ("s_agent2", 4)), n_steps=4
    )
    model_name = "multi_agent_coordination"
    sim_dir = tmp_path / "execution" / model_name / "rxinfer" / "simulation_data"
    sim_dir.mkdir(parents=True)
    results_file = sim_dir / f"{model_name}_simulation_results.json"
    results_file.write_text(json.dumps(results), encoding="utf-8")

    out = tmp_path / "viz"
    paths = generate_analysis_from_logs(tmp_path / "execution", out)
    names = _basenames(paths)

    assert f"{model_name}_rxinfer_animation.gif" in names
    assert f"{model_name}_rxinfer_per_factor_beliefs.png" in names
    for path in paths:
        assert Path(path).exists() and Path(path).stat().st_size > 0

    # generate_analysis_from_logs returns file paths, not the enriched payload, so
    # assert the payload enrichment on the same JSON the pipeline reads.
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    create_rxinfer_visualizations(payload, out, model_name)
    assert set(payload["per_factor_beliefs"]) == {"s_agent1", "s_agent2"}
    assert len(payload["per_factor_beliefs"]["s_agent1"]) == 4


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
