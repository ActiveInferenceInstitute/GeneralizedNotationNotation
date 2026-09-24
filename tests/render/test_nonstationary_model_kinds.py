#!/usr/bin/env python3
"""ModelKind.NONSTATIONARY detection, refusal receipts, and pymdp passthrough.

GEN-4 nonstationary semantics: a time-indexed (``B_t``) or regime-switched
(``B_regime`` + ``b_regime_schedule``) transition parameterization is
detected as NONSTATIONARY from typed fields only, refused with the stable
``unsupported-nonstationary:`` receipt anchor by every render target that
cannot express time variation, and passed through raw (no static-B
canonicalisation) to the pymdp route.
"""

from __future__ import annotations

from pathlib import Path

from gnn.render.pomdp_contract import (
    ModelKind,
    detect_model_kind,
    detect_model_kinds,
    unsupported_nonstationary_reason,
)
from gnn.render.processor import render_gnn_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCRETE_DIR = REPO_ROOT / "input" / "gnn_files" / "discrete"

_A = [
    [0.85, 0.10, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.10, 0.85],
]
_C = [0.0, 0.0, 1.0]
_D = [0.34, 0.33, 0.33]

# (next, prev, action) slices — one per regime / phase.
_SLICE_CALM = [
    [[0.7, 0.1], [0.2, 0.1], [0.1, 0.8]],
    [[0.2, 0.1], [0.7, 0.1], [0.1, 0.8]],
    [[0.1, 0.1], [0.1, 0.1], [0.8, 0.8]],
]
_SLICE_STORM = [
    [[0.2, 0.4], [0.6, 0.2], [0.2, 0.4]],
    [[0.4, 0.2], [0.2, 0.4], [0.4, 0.4]],
    [[0.3, 0.1], [0.3, 0.1], [0.4, 0.8]],
]


def _regime_spec() -> dict:
    return {
        "model_name": "regime-switched",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A": _A,
            "B_regime": [_SLICE_CALM, _SLICE_STORM],
            "C": _C,
            "D": _D,
        },
        "model_parameters": {
            "num_hidden_states": 3,
            "num_obs": 3,
            "num_actions": 2,
            "num_timesteps": 6,
            "b_regime_schedule": [0, 0, 0, 1, 1, 1],
        },
    }


def _time_varying_spec() -> dict:
    return {
        "model_name": "time-varying",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A": _A,
            "B_t": [_SLICE_CALM, _SLICE_STORM],
            "C": _C,
            "D": _D,
        },
        "model_parameters": {
            "num_hidden_states": 3,
            "num_obs": 3,
            "num_actions": 2,
            "num_timesteps": 4,
        },
    }


def _static_spec() -> dict:
    return {
        "model_name": "static",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A": _A,
            "B": _SLICE_CALM,
            "C": _C,
            "D": _D,
        },
        "model_parameters": {
            "num_hidden_states": 3,
            "num_obs": 3,
            "num_actions": 2,
        },
    }


# --- kind detection ----------------------------------------------------------


def test_regime_switched_spec_detects_nonstationary() -> None:
    assert detect_model_kinds(_regime_spec()) == frozenset({ModelKind.NONSTATIONARY})
    assert detect_model_kind(_regime_spec()) is ModelKind.NONSTATIONARY


def test_time_varying_spec_detects_nonstationary() -> None:
    assert detect_model_kinds(_time_varying_spec()) == frozenset(
        {ModelKind.NONSTATIONARY}
    )
    assert detect_model_kind(_time_varying_spec()) is ModelKind.NONSTATIONARY


def test_schedule_parameter_alone_detects_nonstationary() -> None:
    spec = _static_spec()
    spec["model_parameters"]["b_regime_schedule"] = [0, 1, 0]
    assert ModelKind.NONSTATIONARY in detect_model_kinds(spec)


def test_static_spec_does_not_detect_nonstationary() -> None:
    assert ModelKind.NONSTATIONARY not in detect_model_kinds(_static_spec())
    assert detect_model_kind(_static_spec()) is ModelKind.FLAT


def test_nonstationary_is_not_structural() -> None:
    """Precedence contract: a time-varying discrete model is NONSTATIONARY,
    not the STRUCTURAL blanket wrapper."""
    assert detect_model_kind(_time_varying_spec()) is not ModelKind.STRUCTURAL


def test_exemplars_detect_nonstationary() -> None:
    """The committed exemplars carry the detection keys their docs claim."""
    for name in ("time_varying_dynamics.md", "regime_switched_dynamics.md"):
        path = DISCRETE_DIR / name
        content = path.read_text(encoding="utf-8")
        assert "B_t" in content or "B_regime" in content, name


def test_unsupported_nonstationary_reason_anchor() -> None:
    reason = unsupported_nonstationary_reason(frozenset({ModelKind.NONSTATIONARY}))
    assert reason.startswith("unsupported-nonstationary: ")


def test_continuous_composition_keeps_composition_receipt() -> None:
    """A continuous parameterization composed with B_t keeps the existing
    unsupported-composition receipt (the CONTINUOUS gate fires first)."""
    spec = _time_varying_spec()
    spec["initialparameterization"].update(
        {
            "F": [[0.9, 0.0], [0.0, 0.9]],
            "H": [[1.0, 0.0], [0.0, 1.0]],
            "Q": [[0.01, 0.0], [0.0, 0.01]],
            "R": [[0.01, 0.0], [0.0, 0.01]],
            "prior_mean": [0.0, 0.0],
            "prior_cov": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    spec["initialparameterization"].pop("A")
    spec["initialparameterization"].pop("C")
    spec["initialparameterization"].pop("D")
    kinds = detect_model_kinds(spec)
    assert ModelKind.CONTINUOUS in kinds and ModelKind.NONSTATIONARY in kinds


# --- render refusals ---------------------------------------------------------


def test_non_pymdp_render_targets_refuse_nonstationary(tmp_path: Path) -> None:
    for target in ("rxinfer", "discopy", "bnlearn", "stan", "jax"):
        success, message, artifacts = render_gnn_spec(
            _regime_spec(), target, tmp_path / target
        )
        assert not success, target
        assert "unsupported-nonstationary: " in message, (target, message)
        assert artifacts == [], target


def test_rxinfer_renderer_raises_nonstationary_receipt(tmp_path: Path) -> None:
    from gnn.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

    success, message, _artifacts = render_gnn_to_rxinfer(
        _regime_spec(), tmp_path / "model_rxinfer.jl"
    )
    assert not success
    assert "unsupported-nonstationary: " in message


# --- pymdp passthrough -------------------------------------------------------


def test_pymdp_renders_nonstationary_pipeline_runner(tmp_path: Path) -> None:
    """The pymdp route bypasses canonicalisation and embeds the raw tensor."""
    output_file = tmp_path / "regime_pymdp.py"
    success, message, _warnings = render_gnn_spec(
        _regime_spec(), "pymdp", tmp_path, {"output_filename": "regime"}
    )
    assert success, message
    code = output_file.read_text(encoding="utf-8")
    compile(code, output_file.name, "exec")
    assert "B_regime" in code
    assert "run_pymdp_simulation" in code
    assert "b_regime_schedule" in code


def test_pymdp_renders_time_varying_pipeline_runner(tmp_path: Path) -> None:
    output_file = tmp_path / "time_varying_pymdp.py"
    success, message, _warnings = render_gnn_spec(
        _time_varying_spec(), "pymdp", tmp_path, {"output_filename": "time_varying"}
    )
    assert success, message
    code = output_file.read_text(encoding="utf-8")
    compile(code, output_file.name, "exec")
    assert "B_t" in code


def test_pymdp_standalone_mode_refuses_nonstationary(tmp_path: Path) -> None:
    output_file = tmp_path / "regime_standalone_pymdp.py"
    success, message, _warnings = render_gnn_spec(
        _regime_spec(),
        "pymdp",
        tmp_path,
        {"output_filename": "regime", "mode": "standalone"},
    )
    assert not success
    assert "unsupported-nonstationary: " in message
    assert not output_file.exists()


def test_pymdp_gate_receipts_a_t_only_nonstationary() -> None:
    """Time variation outside B has no pymdp executor route: gate receipts it
    instead of silently rolling the spec out as a static model."""
    from gnn.execute.pymdp.simulation import pymdp_kind_refusal

    spec = {
        "model_name": "a-t-only",
        "gnn_section": "ActInfPOMDP",
        "initialparameterization": {
            "A_t": [_A, _A],
            "B": _SLICE_CALM,
            "C": _C,
            "D": _D,
        },
        "model_parameters": {
            "num_hidden_states": 3,
            "num_obs": 3,
            "num_actions": 2,
        },
    }
    receipt = pymdp_kind_refusal(spec)
    assert receipt is not None
    assert receipt["unsupported"] is True
    assert receipt["status"] == "unsupported"
    assert receipt["reason"].startswith("unsupported-nonstationary: ")


def test_pymdp_gate_allows_b_regime_nonstationary() -> None:
    from gnn.execute.pymdp.simulation import pymdp_kind_refusal

    assert pymdp_kind_refusal(_regime_spec()) is None
