"""Factored-continuous (per-factor LGSSM) render contracts and refusals.

The factored-continuous composition is ``kinds == {FACTORED, CONTINUOUS}``
(``num_factors > 1`` alongside a linear-Gaussian family). The per-factor key
convention mirrors the Kronecker ``^([ABCD])_f(\\d+)$`` regex: ``F_fN`` /
``H_fN`` / ``Q_fN`` / ``R_fN`` / ``prior_mean_fN`` / ``prior_cov_fN`` per
1-indexed contiguous factor, with optional ``goal_mean_fN`` +
``control_gain_fN`` (both-or-neither per factor). Only the JAX backend can
express the family: every other target is receipted with the stable
``unsupported-factored-continuous`` prefix, and the HYBRID family mix is
receipted with ``unsupported-composition`` — never silently rendered as one
family with the other dropped. Refusals happen before any renderer import,
so these tests stay free of heavy optional dependencies.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from gnn.render.continuous_common import (
    extract_continuous_spec,
    extract_factored_continuous_spec,
    literal_block_factored,
)
from gnn.render.continuous_script import (
    generate_continuous_script,
    generate_factored_continuous_script,
)
from gnn.render.pomdp_contract import ModelKind, detect_model_kinds
from gnn.render.processor import render_gnn_spec

T = 8

#: Factor 1 declares the optional goal/control pair; factor 2 stays passive,
#: mirroring the factored-continuous exemplar's both-or-neither convention.
_GOAL_FACTOR = 1


def _factored_spec(with_control: bool = False) -> Dict[str, Any]:
    """Two-factor LGSSM spec under the pinned per-factor key convention."""
    initial: Dict[str, Any] = {
        # Factor 1: 2 latent states, 2 observations.
        "F_f1": [[0.9, 0.0], [0.0, 0.8]],
        "H_f1": [[1.0, 0.0], [0.0, 1.0]],
        "Q_f1": [[0.01, 0.0], [0.0, 0.01]],
        "R_f1": [[0.1, 0.0], [0.0, 0.1]],
        "prior_mean_f1": [0.0, 0.0],
        "prior_cov_f1": [[0.5, 0.0], [0.0, 0.5]],
        # Factor 2: 2 latent states, 1 observation (per-factor independence).
        "F_f2": [[1.0, 0.1], [0.0, 0.9]],
        "H_f2": [[1.0, 0.0]],
        "Q_f2": [[0.02, 0.0], [0.0, 0.02]],
        "R_f2": [[0.2]],
        "prior_mean_f2": [0.1, 0.0],
        "prior_cov_f2": [[0.4, 0.0], [0.0, 0.4]],
    }
    if with_control:
        suffix = f"_f{_GOAL_FACTOR}"
        initial[f"goal_mean{suffix}"] = [0.5, 0.5]
        initial[f"control_gain{suffix}"] = 0.25
    return {
        "model_name": "Factored Continuous LGSSM",
        "initialparameterization": initial,
        "model_parameters": {
            "num_factors": 2,
            "num_timesteps": T,
            "dt": 0.1,
            "random_seed": 7,
        },
    }


def _hybrid_spec() -> Dict[str, Any]:
    """Discrete A/B/C/D keys alongside the linear-Gaussian family."""
    return {
        "model_name": "Hybrid Discrete Continuous",
        "initialparameterization": {
            "A": [[0.8, 0.2], [0.1, 0.9]],
            "B": [[[1.0, 0.0], [0.0, 1.0]]],
            "C": [1.0, 0.0],
            "D": [0.5, 0.5],
            "F": [[1.0, 0.0], [0.0, 1.0]],
            "H": [[1.0, 0.0], [0.0, 1.0]],
            "Q": [[0.05, 0.0], [0.0, 0.05]],
            "R": [[0.1, 0.0], [0.0, 0.1]],
            "prior_mean": [0.0, 0.0],
            "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
        },
        "model_parameters": {},
    }


def _flat_continuous_spec() -> Dict[str, Any]:
    """The plain (non-factored) continuous spec for the flat regression."""
    return {
        "model_name": "Flat Continuous LGSSM",
        "initialparameterization": {
            "F": [[0.9, 0.0], [0.0, 0.8]],
            "H": [[1.0, 0.0], [0.0, 1.0]],
            "Q": [[0.01, 0.0], [0.0, 0.01]],
            "R": [[0.1, 0.0], [0.0, 0.1]],
            "prior_mean": [0.0, 0.0],
            "prior_cov": [[0.5, 0.0], [0.0, 0.5]],
        },
        "model_parameters": {"num_timesteps": T, "dt": 0.1, "random_seed": 7},
    }


class TestFactoredExtraction:
    """extract_factored_continuous_spec parses per-factor LGSSM blocks."""

    def test_extract_happy_path(self) -> None:
        spec = extract_factored_continuous_spec(_factored_spec(with_control=True))
        assert spec.num_factors == 2
        assert len(spec.factors) == 2
        factor1, factor2 = spec.factors
        assert factor1.n == 2 and factor1.m == 2
        assert factor2.n == 2 and factor2.m == 1
        assert factor1.has_control is True
        assert factor2.has_control is False
        assert factor1.control_gain == pytest.approx(0.25)
        assert [float(v) for v in factor1.goal_mean] == pytest.approx([0.5, 0.5])

    def test_literal_block_factored_round_trips_shapes(self) -> None:
        """Every per-factor literal parses back to the declared nested shapes."""
        spec = extract_factored_continuous_spec(_factored_spec())
        block = literal_block_factored(spec)
        for key in (
            "F_f1",
            "H_f1",
            "Q_f1",
            "R_f1",
            "prior_mean_f1",
            "prior_cov_f1",
            "F_f2",
            "H_f2",
            "Q_f2",
            "R_f2",
            "prior_mean_f2",
            "prior_cov_f2",
        ):
            assert key in block, key
        assert ast.literal_eval(block["F_f1"]) == [[0.9, 0.0], [0.0, 0.8]]
        assert ast.literal_eval(block["H_f1"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert ast.literal_eval(block["prior_cov_f1"]) == [[0.5, 0.0], [0.0, 0.5]]
        assert ast.literal_eval(block["F_f2"]) == [[1.0, 0.1], [0.0, 0.9]]
        assert ast.literal_eval(block["H_f2"]) == [[1.0, 0.0]]
        assert ast.literal_eval(block["R_f2"]) == [[0.2]]
        assert ast.literal_eval(block["prior_mean_f2"]) == [0.1, 0.0]

    def test_extract_missing_factor_key_raises(self) -> None:
        spec = _factored_spec()
        del spec["initialparameterization"]["R_f2"]
        with pytest.raises(ValueError, match="R_f2"):
            extract_factored_continuous_spec(spec)

    def test_extract_num_factors_below_two_raises(self) -> None:
        spec = _factored_spec()
        spec["model_parameters"]["num_factors"] = 1
        with pytest.raises(ValueError, match="num_factors"):
            extract_factored_continuous_spec(spec)

    def test_extract_goal_without_control_gain_raises(self) -> None:
        """goal_mean_fN without control_gain_fN violates both-or-neither."""
        spec = _factored_spec(with_control=True)
        del spec["initialparameterization"][f"control_gain_f{_GOAL_FACTOR}"]
        with pytest.raises(ValueError, match=f"control_gain_f{_GOAL_FACTOR}"):
            extract_factored_continuous_spec(spec)


class TestFactoredScriptGeneration:
    """generate_factored_continuous_script emits compilable per-factor code."""

    def test_jax_script_embeds_factors_and_compiles(self) -> None:
        spec = extract_factored_continuous_spec(_factored_spec(with_control=True))
        code = generate_factored_continuous_script(spec, "jax")
        assert "factored_continuous" in code
        assert re.search(r"NUM_FACTORS\s*=\s*2\b", code)
        # The JAX output env var is the flat-continuous convention value.
        assert "GNN_OUTPUT_DIR" in code
        for literal in literal_block_factored(spec).values():
            assert literal in code, literal
        # String templating only: the script must compile without importing jax.
        compile(code, "<generated>", "exec")

    def test_unsupported_backend_raises(self) -> None:
        spec = extract_factored_continuous_spec(_factored_spec())
        with pytest.raises(ValueError, match="unsupported factored-continuous backend"):
            generate_factored_continuous_script(spec, "numpyro")


class TestFactoredDispatch:
    """render_gnn_spec routes the factored-continuous carve-out, refusing others."""

    def test_factored_mapping_kinds(self) -> None:
        """The dispatch pre-guard: kinds are exactly {FACTORED, CONTINUOUS}."""
        kinds = detect_model_kinds(_factored_spec())
        assert kinds == frozenset({ModelKind.FACTORED, ModelKind.CONTINUOUS})

    def test_render_gnn_spec_renders_factored_continuous_to_jax(
        self, tmp_path: Path
    ) -> None:
        success, message, files = render_gnn_spec(_factored_spec(), "jax", tmp_path)
        assert success is True, message
        written = [Path(p) for p in files]
        jax_scripts = [p for p in written if p.name.endswith("_jax.py")]
        assert jax_scripts, written
        assert "factored_continuous" in jax_scripts[0].read_text()

    def test_render_gnn_spec_refuses_factored_continuous_for_numpyro(
        self, tmp_path: Path
    ) -> None:
        success, message, files = render_gnn_spec(_factored_spec(), "numpyro", tmp_path)
        assert success is False
        assert files == []
        assert "unsupported-factored-continuous" in message

    @pytest.mark.parametrize("target", ["pymdp", "jax"])
    def test_render_gnn_spec_refuses_hybrid_composition(
        self, target: str, tmp_path: Path
    ) -> None:
        """HYBRID is refused with the stable composition receipt, never rendered."""
        success, message, files = render_gnn_spec(_hybrid_spec(), target, tmp_path)
        assert success is False
        assert files == []
        assert "unsupported-composition" in message
        assert "hybrid" in message

    def test_flat_continuous_script_regression(self) -> None:
        """The flat continuous branch keeps model_kind 'continuous', no FACTORS."""
        spec = extract_continuous_spec(_flat_continuous_spec())
        code = generate_continuous_script(spec, "jax")
        assert '"model_kind": "continuous"' in code
        assert "FACTORS" not in code
