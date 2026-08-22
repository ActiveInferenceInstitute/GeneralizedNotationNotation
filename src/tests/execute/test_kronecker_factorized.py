"""Tests for sparse Kronecker-factorized discrete active inference (MAJ-02).

Pins three invariants of ``execute.jax.kronecker_factorized``:

1. **Kronecker identities** — ``kron_matvec`` and ``kron_matvec_flat`` agree
   with the dense Kronecker product (the joint is only materialised in the
   test, never in the execution path).
2. **Exact EFE decomposition** — the sum of per-factor EFE values equals the
   dense EFE evaluated at a factorised posterior (mean-field exactness for
   factor-separable models).
3. **N >= 64 execution** — factorised models whose joint state space is 64,
   128 or 256 states run to completion with ``joint_materialized: False``
   and valid trajectories.

Plus the MAJ-02 probe surface: the factorised GNN spec generator parses
through the real extractor, and the scaling script's ``--factorized`` sweep
completes runs for N >= 64 states.

Pure JAX/numpy — no Julia, no GPU required, zero skips.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: Path) -> Any:
    scripts_dir = str(relative_path.parent)
    need_cleanup = scripts_dir not in sys.path
    if need_cleanup:
        sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(name, relative_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if need_cleanup and scripts_dir in sys.path:
            sys.path.remove(scripts_dir)


from execute.jax.kronecker_factorized import (
    FactorizedPOMDP,
    build_binary_factor_model,
    build_generic_factor_model,
    factorized_state_space_size,
    kron_materialize,
    kron_matvec,
    kron_matvec_flat,
    per_factor_efe,
    run_factorized_active_inference,
)


def _stochastic(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.random((rows, cols))
    return np.asarray(matrix / matrix.sum(axis=0), dtype=np.float64)


class TestKroneckerIdentities:
    """Sparse applications agree with the dense Kronecker product."""

    def test_factorized_matvec_matches_dense(self) -> None:
        rng = np.random.default_rng(0)
        factors = [
            _stochastic(rng, 3, 3),
            _stochastic(rng, 2, 2),
            _stochastic(rng, 4, 4),
        ]
        vectors = [
            rng.random(3) + 0.5,
            rng.random(2) + 0.5,
            rng.random(4) + 0.5,
        ]
        dense = kron_materialize(factors)
        dense_vector = np.kron(np.kron(vectors[0], vectors[1]), vectors[2])

        applied = kron_matvec(factors, vectors)
        # kron_matvec normalises each factor output (distribution semantics);
        # normalising the dense product factor-by-factor is identical.
        for factor_result, vector, matrix in zip(applied, vectors, factors):
            expected = matrix @ vector
            expected = expected / expected.sum()
            assert np.allclose(np.asarray(factor_result), expected)
        assert np.allclose(
            np.asarray(kron_matvec_flat(factors, dense_vector)), dense @ dense_vector
        )

    def test_flat_matvec_accepts_grid_input(self) -> None:
        rng = np.random.default_rng(1)
        factors = [_stochastic(rng, 2, 2), _stochastic(rng, 3, 3)]
        grid = rng.random((2, 3))
        dense = kron_materialize(factors)
        flat = grid.ravel()
        assert np.allclose(
            np.asarray(kron_matvec_flat(factors, grid)),
            dense @ flat,
        )

    def test_single_factor_is_ordinary_matvec(self) -> None:
        rng = np.random.default_rng(2)
        matrix = _stochastic(rng, 5, 5)
        vector = rng.random(5) + 0.5
        assert np.allclose(
            np.asarray(kron_matvec_flat([matrix], vector)), matrix @ vector
        )

    def test_flat_matvec_rejects_wrong_length(self) -> None:
        factors = [np.eye(2), np.eye(3)]
        with pytest.raises(ValueError, match="does not match the joint size"):
            kron_matvec_flat(factors, np.ones(5))

    def test_materialize_shape(self) -> None:
        factors = [np.eye(2), np.eye(3), np.eye(4)]
        assert kron_materialize(factors).shape == (24, 24)


class TestEfeDecomposition:
    """Sum of per-factor EFE equals the dense EFE at a factorised posterior."""

    @pytest.mark.parametrize("factor_sizes", [(2, 2), (3, 2), (3, 2, 4)])
    def test_per_factor_efe_sums_to_dense(self, factor_sizes: tuple) -> None:
        rng = np.random.default_rng(3)
        model = build_generic_factor_model(list(factor_sizes), t=3, seed=1)
        a_matrices = [np.asarray(m) for m in model.A]
        b_matrices = [np.asarray(m) for m in model.B]
        c_prefs = [
            np.asarray(
                np.exp(np.asarray(c)) / np.exp(np.asarray(c)).sum(), dtype=np.float64
            )
            for c in model.C
        ]
        beliefs = [
            np.asarray(rng.random(n) + 0.5, dtype=np.float64)
            for n in model.factor_sizes
        ]
        beliefs = [np.asarray(b / b.sum(), dtype=np.float64) for b in beliefs]

        a_dense = np.asarray(kron_materialize(a_matrices))
        b_dense = np.asarray(kron_materialize(b_matrices))
        q_flat: np.ndarray = beliefs[0]
        for belief in beliefs[1:]:
            q_flat = np.kron(q_flat, belief)
        c_flat: np.ndarray = c_prefs[0]
        for pref in c_prefs[1:]:
            c_flat = np.kron(c_flat, pref)

        # Joint action index in C-order kron layout:
        # index = u_1*(a_2*a_3) + u_2*a_3 + u_3
        action = 1
        index = 0
        for f, a_size in enumerate(reversed(model.action_sizes)):
            index += action * int(np.prod(model.action_sizes[f + 1 :]))

        def dense_efe(belief: np.ndarray, action_index: int) -> float:
            predicted_state = b_dense[:, :, action_index] @ belief
            predicted_state = np.maximum(predicted_state, 1e-16)
            predicted_state /= predicted_state.sum()
            predicted_obs = a_dense @ predicted_state
            predicted_obs = np.maximum(predicted_obs, 1e-16)
            predicted_obs /= predicted_obs.sum()
            ambiguity = 0.0
            for state in range(predicted_state.shape[0]):
                likelihood = np.maximum(a_dense[:, state], 1e-16)
                ambiguity -= predicted_state[state] * float(
                    np.sum(likelihood * np.log(likelihood))
                )
            preferred = np.maximum(c_flat, 1e-16)
            risk = float(
                np.sum(predicted_obs * (np.log(predicted_obs) - np.log(preferred)))
            )
            return ambiguity + risk

        dense_value = dense_efe(q_flat, index)
        factor_sum = sum(
            per_factor_efe(beliefs[f], action, a_matrices[f], b_matrices[f], c_prefs[f])
            for f in range(len(factor_sizes))
        )
        assert dense_value == pytest.approx(factor_sum, abs=1e-6)


class TestFactorizedExecution:
    """The mean-field simulation runs over factorised matrices (N >= 64)."""

    @pytest.mark.parametrize(
        "factor_sizes",
        [([2, 2, 2, 2, 2, 2]), ([4, 4, 4]), ([8, 8])],
    )
    def test_n64_plus_executes_without_joint_materialization(
        self, factor_sizes: list
    ) -> None:
        model = build_generic_factor_model(factor_sizes, t=5, seed=7)
        assert model.joint_state_space_size >= 64
        results = run_factorized_active_inference(model)
        assert results["success"] is True
        assert results["model_kind"] == "factorized_kronecker"
        assert results["model_parameters"]["joint_state_space_size"] >= 64
        assert results["model_parameters"]["joint_materialized"] is False
        assert results["validation"]["all_valid"] is True
        assert len(results["factors"]) == len(factor_sizes)
        for name in results["factors"]:
            assert len(results["actions_by_factor"][name]) == model.T
            assert len(results["beliefs_by_factor"][name]) == model.T

    def test_binary_model_reaches_256_states(self) -> None:
        model = build_binary_factor_model(8, t=5, seed=1)
        assert model.joint_state_space_size == 256
        results = run_factorized_active_inference(model)
        assert results["validation"]["all_valid"] is True
        assert results["model_parameters"]["joint_materialized"] is False

    def test_deterministic_with_seed(self) -> None:
        first = run_factorized_active_inference(
            build_generic_factor_model([3, 2], t=4, seed=11)
        )
        second = run_factorized_active_inference(
            build_generic_factor_model([3, 2], t=4, seed=11)
        )
        assert first["actions_by_factor"] == second["actions_by_factor"]
        assert first["beliefs_by_factor"] == second["beliefs_by_factor"]

    def test_validation_flags(self) -> None:
        model = build_generic_factor_model([2, 3], t=5, seed=3)
        results = run_factorized_active_inference(model)
        validation = results["validation"]
        assert validation["all_beliefs_valid"] is True
        assert validation["beliefs_sum_to_one"] is True
        assert validation["actions_in_range"] is True
        # Every belief is a probability vector over the factor's states.
        for name in results["factors"]:
            for belief in results["beliefs_by_factor"][name]:
                assert abs(sum(belief) - 1.0) < 1e-6


class TestModelValidation:
    """Constructor and size guards."""

    def test_factor_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="factor count mismatch"):
            FactorizedPOMDP(
                A=[np.eye(2)],
                B=[np.zeros((2, 2, 2))],
                C=[np.zeros(2)],
                D=[np.zeros(2), np.zeros(2)],
            )

    def test_bad_b_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match state size"):
            FactorizedPOMDP(
                A=[np.eye(2)],
                B=[np.zeros((3, 3, 2))],
                C=[np.zeros(2)],
                D=[np.zeros(2)],
            )

    def test_empty_factor_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one factor"):
            FactorizedPOMDP(A=[], B=[], C=[], D=[])

    def test_state_space_size_helper(self) -> None:
        assert factorized_state_space_size([2, 3, 4]) == 24
        assert factorized_state_space_size([]) == 1


class TestFactorizedSpecGenerator:
    """The MAJ-02 GNN spec generator produces parseable factor-separable specs."""

    def test_spec_parses_with_per_factor_matrices(self) -> None:
        from gnn.pomdp_extractor import extract_pomdp_from_content

        generator = _load_module(
            "pymdp_spec_generator",
            PROJECT_ROOT / "scripts" / "pymdp_spec_generator.py",
        )
        content = generator.generate_factorized_gnn_file([3, 4], 10)
        assert "Kronecker Factorized N12 T10" in content
        pomdp = extract_pomdp_from_content(content, strict_validation=True)
        assert pomdp is not None
        matrices = getattr(pomdp, "matrices", None) or {}
        assert sorted(matrices) == [
            "A_f0",
            "A_f1",
            "B_f0",
            "B_f1",
            "C_f0",
            "C_f1",
            "D_f0",
            "D_f1",
        ]
        factors = getattr(pomdp, "state_factors", None) or []
        assert [int(f["size"]) for f in factors] == [3, 4]


class TestScalingSweep:
    """The scaling script's ``--factorized`` sweep (MAJ-02 probe)."""

    def test_factorized_sweep_completes_for_n64(self, tmp_path: Path) -> None:
        scaling = _load_module(
            "pymdp_scaling",
            PROJECT_ROOT / "scripts" / "run_pymdp_gnn_scaling_analysis.py",
        )
        factor_out = tmp_path / "specs"
        pipeline_out = tmp_path / "output"
        args = SimpleNamespace(
            factors="4,4,4",
            factor_timesteps="5",
            factor_output_dir=str(factor_out),
            pipeline_output_dir=str(pipeline_out),
            a_signal=0.85,
        )
        assert scaling._run_factorized_sweep(args) == 0

        manifest_path = pipeline_out / "pymdp_kronecker_scaling_manifest.json"
        assert manifest_path.exists()
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "pymdp_kronecker_scaling_manifest_v1"
        assert manifest["joint_state_space_size"] == 64
        assert manifest["runs"][0]["joint_materialized"] is False
        assert manifest["runs"][0]["validation_all_valid"] is True
        assert (factor_out / "kronecker_N64_T5.md").exists()


class TestKroneckerPipelineExecutor:
    """The Step-12-routable Kronecker executor (MAJ-02 residual).

    ``execute/ax.kronecker_executor`` turns a factorised model into a
    ``simulation_results.json`` carrying the ``jax_kronecker_factorized_v1``
    schema in the standard ``simulation_data/`` location, and exposes it from
    ``execute.jax`` (the public jax submodule surface) so the numbered
    pipeline can route factorised Kronecker execution through Step 12.
    """

    import json as _json

    def test_executor_writes_schema_artifact(self, tmp_path: Path) -> None:
        from execute.jax.kronecker_executor import execute_kronecker_factorized

        envelope = execute_kronecker_factorized(
            {"factor_sizes": [4, 4, 4], "t": 5, "seed": 7}, tmp_path
        )
        assert envelope["success"] is True
        assert envelope["schema_version"] == "jax_kronecker_factorized_v1"

        results_file = tmp_path / "simulation_data" / "simulation_results.json"
        assert results_file.exists(), "schema artifact was not written"
        results = self._json.loads(results_file.read_text(encoding="utf-8"))
        assert results["schema_version"] == "jax_kronecker_factorized_v1"
        assert results["model_parameters"]["joint_state_space_size"] >= 64
        assert results["model_parameters"]["joint_materialized"] is False
        assert results["validation"]["all_valid"] is True

        summary_file = tmp_path / "kronecker_execution_summary.json"
        assert summary_file.exists()
        summary = self._json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary["schema_version"] == "jax_kronecker_factorized_v1"
        assert summary["joint_materialized"] is False
        assert summary["all_valid"] is True

    def test_executor_dispatches_binary_vs_generic_builders(self, tmp_path: Path) -> None:
        from execute.jax.kronecker_executor import (
            _build_factor_model,
            execute_kronecker_factorized,
        )

        # Homogeneous binary factors use the binary builder; mixed sizes use generic.
        binary = _build_factor_model([2, 2, 2, 2, 2, 2], t=3, seed=1)
        generic = _build_factor_model([3, 2, 4], t=3, seed=1)
        assert binary.joint_state_space_size == 64
        assert generic.joint_state_space_size == 24

        mixed = execute_kronecker_factorized(
            {"factor_sizes": [3, 2, 4], "t": 3, "seed": 5}, tmp_path
        )
        assert mixed["success"] is True
        results = self._json.loads(
            (tmp_path / "simulation_data" / "simulation_results.json").read_text(
                encoding="utf-8"
            )
        )
        assert results["model_parameters"]["joint_state_space_size"] == 24

    def test_executor_exported_from_jax_surface(self) -> None:
        from execute.jax import __all__ as jax_all
        from execute.jax.kronecker_executor import (
            execute_kronecker_factorized as exported_execute,
        )

        assert "execute_kronecker_factorized" in jax_all
        assert "run_kronecker_factorized_execution" in jax_all
        assert callable(exported_execute)

    def test_config_validation(self) -> None:
        from execute.jax.kronecker_executor import execute_kronecker_factorized

        with pytest.raises(ValueError, match="dict"):
            execute_kronecker_factorized(["not-a-dict"], "/tmp/unused")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="factor_sizes"):
            execute_kronecker_factorized({}, "/tmp/unused")
