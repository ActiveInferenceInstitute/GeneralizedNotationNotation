"""Tests for EFE (Expected Free Energy) formula correctness in the generated RxInfer code.

Validates that the EFE computation embedded in the generated Julia scripts
has the correct structure and sign conventions by inspecting the rendered
source code. This addresses the finding that the EFE formula was unvalidated.

The EFE formula in the generated code is:
  EFE(action) = ambiguity + risk
where:
  ambiguity = -sum_s q(s) * sum_o A[o,s] * log(A[o,s])   (expected entropy of likelihood)
  risk = sum_o q(o|action) * (log q(o|action) - log C_pref)  (KL divergence from preference)

Sign convention: lower EFE is better (selected via softmax(-precision * EFE)).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

REPO_ROOT = Path(__file__).resolve().parents[3]
SIMPLE_MDP = REPO_ROOT / "input" / "gnn_files" / "discrete" / "simple_mdp.md"


def _render_simple_mdp(tmp_path: Path) -> Path:
    """Render the simple_mdp exemplar and return the generated Julia script path."""
    assert SIMPLE_MDP.exists(), f"missing exemplar: {SIMPLE_MDP}"
    pomdp_space = extract_pomdp_from_file(SIMPLE_MDP, strict_validation=True)
    assert pomdp_space is not None
    gnn_spec = POMDPRenderProcessor(tmp_path)._pomdp_to_gnn_spec(pomdp_space)
    output_path = tmp_path / "simple_mdp_rxinfer.jl"
    success, message, _ = render_gnn_to_rxinfer(gnn_spec, output_path)
    assert success, f"render failed: {message}"
    assert output_path.exists()
    return output_path


def test_generated_code_contains_efe_function(tmp_path: Path) -> None:
    """The generated Julia code must define compute_efe with the correct signature."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    assert "function compute_efe(" in source, "compute_efe function missing"
    assert "ambiguity" in source, "ambiguity term missing from EFE"
    assert "risk" in source, "risk term missing from EFE"


def test_efe_uses_correct_sign_convention(tmp_path: Path) -> None:
    """EFE must use the convention: lower EFE is better.

    Action selection is softmax(log E - ACTION_PRECISION * efe_values): the
    habit prior E enters via log-add (uniform E cancels inside softmax) and
    EFE must be NEGATED (lower EFE = better action), never added.
    """
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    # select_action and compute_efe_and_policy must negate EFE, with the
    # habit prior as a log-additive term.
    assert (
        "softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)"
        in source
    ), (
        "Action selection must use softmax(log E - ACTION_PRECISION * efe) "
        "(lower EFE = better action, habit prior log-added)"
    )
    assert "+ ACTION_PRECISION .* efe_values" not in source


def test_efe_ambiguity_uses_log_likelihood(tmp_path: Path) -> None:
    """Ambiguity term must use log(A[:, state]) — the negative entropy of the likelihood."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    # The ambiguity computation: -sum_s q(s) * sum_o A[o,s] * log(A[o,s])
    assert "log.(likelihood)" in source or "log.(A[:, state])" in source, (
        "Ambiguity must compute log-likelihood entropy"
    )


def test_efe_risk_uses_kl_divergence(tmp_path: Path) -> None:
    """Risk term must compute KL divergence: sum q(o) * (log q(o) - log C_pref)."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    # Risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(preferred)))
    assert "log.(predicted_obs)" in source, (
        "Risk must use log of predicted observations"
    )
    assert "log.(preferred)" in source, "Risk must use log of preferences (C)"
    assert ".-" in source or " .- " in source, "Risk must compute log difference (KL)"


def test_efe_predicted_state_from_b_transition(tmp_path: Path) -> None:
    """EFE must predict next state using B[:, :, action] * belief."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    assert "B[:, :, action] * belief" in source, (
        "EFE must use B transition to predict next state"
    )


def test_efe_predicted_obs_from_a_likelihood(tmp_path: Path) -> None:
    """EFE must predict observations using A * predicted_state."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    assert "A * predicted_state" in source, (
        "EFE must use A likelihood to predict observations"
    )


def test_efe_normalizes_predictions(tmp_path: Path) -> None:
    """EFE must normalize predicted state and observations to valid distributions."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    # Check for normalization after prediction
    assert "predicted_state ./= sum(predicted_state)" in source or (
        "predicted_state = predicted_state ./ sum(predicted_state)" in source
    ), "Predicted state must be normalized"
    assert "predicted_obs ./= sum(predicted_obs)" in source or (
        "predicted_obs = predicted_obs ./ sum(predicted_obs)" in source
    ), "Predicted observations must be normalized"


def test_efe_preference_from_c_vector(tmp_path: Path) -> None:
    """EFE must use softmax(C) as the preference distribution."""
    source = _render_simple_mdp(tmp_path).read_text(encoding="utf-8")
    assert "C_pref = softmax(C)" in source, "Preference must be softmax(C)"
