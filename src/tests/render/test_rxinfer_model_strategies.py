"""Regression tests for RxInfer ModelKind detection + strategy dispatch.

Pins two invariants the 2026-08-05 red-team review found broken:

1. ``detect_model_kind`` is STRUCTURAL — prose in a ModelName or annotation
   must never change how a model renders (the old ``str(gnn_spec)``
   substring scan misrouted ``temporal_hierarchy.md`` on the word
   "Hierarchy" in its name and made every exemplar one doc-comment away
   from a render failure).
2. Every GNN exemplar renders through the real pipeline path (the
   "29/29 render" contract), with the intended kind taxonomy.

Pure Python — no Julia required, zero skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_contract import (
    ModelKind,
    build_canonical_pomdp_spec,
    detect_model_kind,
)
from render.pomdp_processor import pomdp_to_gnn_spec
from render.rxinfer.model_strategies import (
    ContinuousStrategy,
    FactoredStrategy,
    FlatStrategy,
    HierarchicalStrategy,
    LearningStrategy,
    MultiAgentStrategy,
    get_model_strategy,
)
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GNN_FILES = PROJECT_ROOT / "input" / "gnn_files"

EXEMPLAR_COUNT = 29

# The intended kind for every non-flat exemplar; everything else is FLAT.
EXPECTED_NON_FLAT = {
    "continuous/continuous_navigation.md": ModelKind.CONTINUOUS,
    "continuous/predictive_coding_agent.md": ModelKind.CONTINUOUS,
    "continuous/stochastic_dynamics.md": ModelKind.CONTINUOUS,
    "hierarchical/hierarchical_pomdp.md": ModelKind.HIERARCHICAL,
    "hierarchical/temporal_hierarchy.md": ModelKind.HIERARCHICAL,
    "learning/dirichlet_likelihood_learning.md": ModelKind.LEARNING,
    "multiagent/multi_agent_coordination.md": ModelKind.MULTI_AGENT,
    "multiagent/stigmergic_swarm.md": ModelKind.MULTI_AGENT,
    "structured/factorized_posterior.md": ModelKind.FACTORED,
}


def _exemplar_files() -> list:
    files = [
        f
        for f in sorted(GNN_FILES.rglob("*.md"))
        if f.name not in ("README.md", "AGENTS.md")
    ]
    assert len(files) == EXEMPLAR_COUNT, (
        f"expected {EXEMPLAR_COUNT} exemplars, found {len(files)}"
    )
    return files


def _canonical_spec(gnn_file: Path) -> dict:
    pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
    assert pomdp is not None, f"extraction failed for {gnn_file}"
    return build_canonical_pomdp_spec(pomdp_to_gnn_spec(pomdp))


class TestExemplarKindTaxonomy:
    """Every exemplar detects its intended kind through the real path."""

    def test_all_exemplars_detect_expected_kind(self) -> None:
        mismatches = []
        for gnn_file in _exemplar_files():
            rel = str(gnn_file.relative_to(GNN_FILES))
            expected = EXPECTED_NON_FLAT.get(rel, ModelKind.FLAT)
            actual = detect_model_kind(_canonical_spec(gnn_file))
            if actual != expected:
                mismatches.append(
                    f"{rel}: expected {expected.value}, got {actual.value}"
                )
        assert not mismatches, "kind misdetections:\n" + "\n".join(mismatches)

    def test_all_exemplars_render(self, tmp_path: Path) -> None:
        """The full-corpus render contract, through the public renderer entry.

        Every kind now renders — natively for flat / hierarchical two-level /
        factored / continuous / learning, and via the documented joint
        composition for multi-agent and 3+-level hierarchical.
        """
        failures = []
        for gnn_file in _exemplar_files():
            pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
            assert pomdp is not None, f"extraction failed for {gnn_file}"
            spec = pomdp_to_gnn_spec(pomdp)
            script = tmp_path / f"{gnn_file.stem}_rxinfer.jl"
            success, message, _warnings = render_gnn_to_rxinfer(spec, script)
            if not success:
                failures.append(f"{gnn_file.name}: {message}")
            else:
                assert script.exists() and script.stat().st_size > 0
        assert not failures, "render failures:\n" + "\n".join(failures)


class TestStructuralDetection:
    """detect_model_kind reads typed fields only — never prose."""

    _BASE_INITIAL = {
        "A": [[1.0, 0.0], [0.0, 1.0]],
        "B": [[[1.0], [0.0]], [[0.0], [1.0]]],
        "C": [0.0, 1.0],
        "D": [0.5, 0.5],
    }

    def test_prose_mentions_do_not_reroute(self) -> None:
        """The exact words that used to misroute renders are now inert."""
        spec = {
            "model_name": "A Hierarchy of Multi_Agent Dirichlet Learning Models",
            "description": (
                "hierarchy multi_agent dirichlet stochastic_dynamics learning"
            ),
            "model_parameters": {},
            "initialparameterization": dict(self._BASE_INITIAL),
        }
        assert detect_model_kind(spec) == ModelKind.FLAT

    def test_agent_metadata_key_does_not_misroute(self) -> None:
        """A scalar 'agent_*' metadata key is not an agent declaration."""
        initial = dict(self._BASE_INITIAL)
        initial["agent_note"] = "single agent model"
        spec = {"initialparameterization": initial, "model_parameters": {}}
        assert detect_model_kind(spec) == ModelKind.FLAT

    def test_non_mapping_initialparameterization_raises(self) -> None:
        """No silent iteration over lists/strings — fail loud."""
        for bad in (["agent_config", "B"], "agent", [("A", 1)], 42):
            with pytest.raises(ValueError, match="must be a mapping"):
                detect_model_kind({"initialparameterization": bad})

    def test_nr_agents_one_stays_flat(self) -> None:
        initial = dict(self._BASE_INITIAL)
        initial["nr_agents"] = 1
        spec = {"initialparameterization": initial, "model_parameters": {}}
        assert detect_model_kind(spec) == ModelKind.FLAT

    def test_agent_matrix_keys_in_structured_pomdp_detect_multi_agent(self) -> None:
        spec = {
            "initialparameterization": dict(self._BASE_INITIAL),
            "model_parameters": {},
            "structured_pomdp": {
                "matrices": {"A_agent1": [[1.0]], "A_agent2": [[1.0]]}
            },
        }
        assert detect_model_kind(spec) == ModelKind.MULTI_AGENT

    def test_level_matrix_keys_detect_hierarchical(self) -> None:
        spec = {
            "initialparameterization": dict(self._BASE_INITIAL),
            "model_parameters": {},
            "structured_pomdp": {
                "matrices": {"A_level1": [[1.0]], "A_level2": [[1.0]]}
            },
        }
        assert detect_model_kind(spec) == ModelKind.HIERARCHICAL

    def test_continuous_parameterization_keys_detect_continuous(self) -> None:
        initial = dict(self._BASE_INITIAL)
        initial.update({"F": [[1.0]], "H": [[1.0]], "Q": [[0.1]], "R": [[0.1]]})
        spec = {"initialparameterization": initial, "model_parameters": {}}
        assert detect_model_kind(spec) == ModelKind.CONTINUOUS


class TestStrategyDispatchAndCodegen:
    """Strategies stamp their own kind and generate runnable-shaped code."""

    def test_joint_composition_strategies_inherit_flat_codegen(self) -> None:
        """Multi-agent keeps the joint composition as its fallback path.

        MultiAgentStrategy renders natively (per-agent + shared env trace)
        when the spec declares >= 2 complete agent groups, and falls back to
        the documented joint composition otherwise — both through the flat
        codegen lineage. FactoredStrategy went native (D3): it must NOT reuse
        the flat codegen.
        """
        assert isinstance(get_model_strategy(ModelKind.FACTORED), FactoredStrategy)
        assert isinstance(get_model_strategy(ModelKind.MULTI_AGENT), MultiAgentStrategy)
        assert isinstance(MultiAgentStrategy(), FlatStrategy)
        # FactoredStrategy went native (D3): it must NOT reuse the flat codegen.
        assert not isinstance(FactoredStrategy(), FlatStrategy)

    def test_every_kind_has_a_registered_strategy(self) -> None:
        for kind in ModelKind:
            assert get_model_strategy(kind).kind is kind

    def test_multi_agent_script_stamps_true_kind_and_echoes_factors(self) -> None:
        gnn_file = GNN_FILES / "multiagent" / "multi_agent_coordination.md"
        code = MultiAgentStrategy().generate_model_code(
            _canonical_spec(gnn_file), "multi_agent_coordination"
        )
        assert 'const MODEL_KIND = "multi_agent"' in code
        assert '"state_factors"' in code

    def test_flat_script_contains_habit_prior_policy(self) -> None:
        """D2: E enters action selection via log-add."""
        gnn_file = GNN_FILES / "discrete" / "actinf_pomdp_agent.md"
        code = FlatStrategy().generate_model_code(
            _canonical_spec(gnn_file), "actinf_pomdp_agent"
        )
        assert "log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values" in code
        assert '"E" => E' in code

    def test_hierarchical_two_level_generates_native_script(self) -> None:
        gnn_file = GNN_FILES / "hierarchical" / "hierarchical_pomdp.md"
        code = HierarchicalStrategy().generate_model_code(
            _canonical_spec(gnn_file), "hierarchical_pomdp"
        )
        assert "hierarchical_pomdp_model" in code
        assert "hierarchical_constraints()" in code
        assert "hierarchical_initialization(NUM_FAST, NUM_SLOW)" in code
        assert 'const MODEL_KIND = "hierarchical"' in code
        assert '"slow_context"' in code

    def test_hierarchical_three_level_renders_joint_composition(self) -> None:
        """3+ declared levels use the documented joint-composition path."""
        gnn_file = GNN_FILES / "hierarchical" / "temporal_hierarchy.md"
        code = HierarchicalStrategy().generate_model_code(
            _canonical_spec(gnn_file), "temporal_hierarchy"
        )
        assert 'const MODEL_KIND = "hierarchical"' in code
        # Joint composition means the flat pomdp_model, not the native chain.
        assert "using GnnRxInferModels: pomdp_model" in code

    def test_hierarchical_missing_level_matrices_raises(self) -> None:
        spec = {
            "model_parameters": {},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
            "structured_pomdp": {
                "matrices": {"A_level1": [[1.0]], "A_level2": [[1.0]]}
            },
        }
        with pytest.raises(ValueError, match="missing per-level"):
            HierarchicalStrategy().generate_model_code(spec, "broken_hierarchical")

    def test_hierarchical_validation_fields_extend_flat(self) -> None:
        fields = HierarchicalStrategy().get_validation_fields()
        assert "context_beliefs_valid" in fields
        assert "context_beliefs_sum_to_one" in fields
        assert "belief_accuracy" in fields


class TestFactoredNativeCodegen:
    """D3: two-factor exemplars render the native mean-field chain."""

    def _code(self) -> str:
        gnn_file = GNN_FILES / "structured" / "factorized_posterior.md"
        return FactoredStrategy().generate_model_code(
            _canonical_spec(gnn_file), "factorized_posterior"
        )

    def test_uses_the_native_factored_model(self) -> None:
        code = self._code()
        assert "factored_pomdp_model" in code
        assert "factored_constraints()" in code
        assert "factored_initialization(N_F0, N_F1)" in code
        # The joint composition's flat model must be gone.
        assert "using GnnRxInferModels: pomdp_model" not in code

    def test_stamps_kind_and_real_factor_names(self) -> None:
        code = self._code()
        assert 'const MODEL_KIND = "factored"' in code
        assert 'const FACTOR0_NAME = "s_f0"' in code
        assert 'const FACTOR1_NAME = "s_f1"' in code
        assert '"beliefs_by_factor" => Dict(' in code
        assert '"posterior_family" => "mean_field_factorized"' in code

    def test_loads_per_factor_matrices_not_the_joint(self) -> None:
        code = self._code()
        for key in ("A_m0", "A_m1", "B_f0", "B_f1", "D_f0", "D_f1"):
            assert f'matrices["{key}"]' in code
        assert 'const B_TENSOR_ORDER = "next_state_previous_state_action"' in code

    def test_validation_fields_cover_both_factors(self) -> None:
        fields = FactoredStrategy().get_validation_fields()
        assert "beliefs_sum_to_one" in fields
        assert "factor1_beliefs_valid" in fields
        assert "factor1_beliefs_sum_to_one" in fields

    def test_missing_per_factor_matrices_raises(self) -> None:
        spec = {
            "model_parameters": {"num_factors": 2, "num_modalities": 2},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
            "structured_pomdp": {"matrices": {"A_m0": [[[1.0]]]}},
        }
        with pytest.raises(ValueError, match="cannot render natively"):
            FactoredStrategy().generate_model_code(spec, "broken_factored")

    def test_wrong_factor_count_raises(self) -> None:
        """Three factors is not the two-factor native path — fail loud."""
        gnn_file = GNN_FILES / "structured" / "factorized_posterior.md"
        spec = _canonical_spec(gnn_file)
        spec["model_parameters"]["num_factors"] = 3
        with pytest.raises(ValueError, match="num_factors is 3"):
            FactoredStrategy().generate_model_code(spec, "factorized_posterior")


class TestContinuousNativeCodegen:
    """A2: continuous exemplars render the linear-Gaussian state-space model."""

    def _code(self, stem: str = "continuous_navigation") -> str:
        gnn_file = GNN_FILES / "continuous" / f"{stem}.md"
        return ContinuousStrategy().generate_model_code(_canonical_spec(gnn_file), stem)

    def test_uses_the_native_continuous_model(self) -> None:
        code = self._code()
        assert "using GnnRxInferModels: continuous_pomdp_model" in code
        assert "continuous_pomdp_model(F = F, H = H, Q = Q, R = R," in code
        # Fully conjugate: no constraints/initialization are needed or passed.
        assert "constraints =" not in code
        assert "initialization =" not in code

    def test_stamps_kind_and_parameterization(self) -> None:
        code = self._code()
        assert 'const MODEL_KIND = "continuous"' in code
        assert '"parameterization" => "linear_gaussian_state_space"' in code
        assert "posterior_cov" in code
        assert '"true_states_continuous" => true_states_continuous' in code

    def test_validation_uses_finiteness_not_positive_free_energy(self) -> None:
        """Continuous Bethe FE is routinely negative — vfe > 0 would be wrong."""
        code = self._code()
        assert "all(isfinite, vfe_per_iteration)" in code
        assert "all(v -> v > 0, vfe_per_iteration)" not in code
        assert "posterior_cov_psd" in code
        assert "rmse_vs_true" in code

    def test_emits_no_fabricated_policy_data(self) -> None:
        code = self._code()
        assert '"efe_per_action" => Vector{Vector{Float64}}(),' in code
        assert '"policy_posterior" => Vector{Vector{Float64}}(),' in code

    def test_all_three_continuous_exemplars_render(self) -> None:
        for stem in (
            "continuous_navigation",
            "predictive_coding_agent",
            "stochastic_dynamics",
        ):
            assert 'const MODEL_KIND = "continuous"' in self._code(stem)

    def test_validation_fields(self) -> None:
        fields = ContinuousStrategy().get_validation_fields()
        assert fields == [
            "vfe_finite",
            "means_finite",
            "posterior_cov_psd",
            "inference_converged",
            "rmse_vs_true",
            "rmse_finite",
        ]

    def test_missing_continuous_parameterization_raises(self) -> None:
        spec = {
            "model_parameters": {},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
        }
        with pytest.raises(ValueError, match="missing the continuous parameterization"):
            ContinuousStrategy().generate_model_code(spec, "no_lgssm")

    def test_partial_continuous_parameterization_names_the_gap(self) -> None:
        initial = dict(TestStructuralDetection._BASE_INITIAL)
        initial.update({"F": [[1.0]], "H": [[1.0]]})
        spec = {"model_parameters": {}, "initialparameterization": initial}
        with pytest.raises(ValueError) as excinfo:
            ContinuousStrategy().generate_model_code(spec, "partial_lgssm")
        message = str(excinfo.value)
        assert "'Q'" in message and "'R'" in message
        assert "'F'" not in message


class TestLearningNativeCodegen:
    """D1: dirichlet_A exemplars render the latent-likelihood model."""

    def _code(self) -> str:
        gnn_file = GNN_FILES / "learning" / "dirichlet_likelihood_learning.md"
        return LearningStrategy().generate_model_code(
            _canonical_spec(gnn_file), "dirichlet_likelihood_learning"
        )

    def test_uses_the_native_learning_model(self) -> None:
        code = self._code()
        assert "learning_pomdp_model" in code
        assert "learning_constraints()" in code
        assert "learning_initialization(prior_counts, NUM_STATES)" in code
        assert "limit_stack_depth = 500" in code

    def test_stamps_kind_and_learned_parameters(self) -> None:
        code = self._code()
        assert 'const MODEL_KIND = "learning"' in code
        assert '"learned_parameters" => ["A"]' in code
        assert 'initial["dirichlet_A"]' in code
        assert '"learned_A_mean" => matrix_rows(A_learned_mean)' in code

    def test_agent_acts_on_the_prior_mean_not_the_true_likelihood(self) -> None:
        """The environment uses true A; the agent uses its Dirichlet belief."""
        code = self._code()
        assert "A_prior_mean = prior_counts ./ sum(prior_counts, dims = 1)" in code
        assert "observation = categorical_index(A_true[:, current_state])" in code
        assert "select_action(current_belief, A_prior_mean, B, C_pref, E)" in code

    def test_learning_is_a_hard_gate(self) -> None:
        code = self._code()
        assert "a_distance_prior" in code
        assert "a_distance_posterior" in code
        assert 'validation["a_learning_improved"]' in code

    def test_vfe_present_means_finite_for_dirichlet_models(self) -> None:
        code = self._code()
        assert "vfe_present = !isempty(vfe_per_iteration) && all(isfinite," in code
        assert "all(v -> v > 0, vfe_per_iteration)" not in code

    def test_validation_fields(self) -> None:
        fields = LearningStrategy().get_validation_fields()
        assert "a_learning_improved" in fields
        assert "a_posterior_columns_normalized" in fields
        assert "belief_accuracy" in fields

    def test_missing_dirichlet_counts_raises(self) -> None:
        spec = {
            "model_parameters": {},
            "initialparameterization": dict(TestStructuralDetection._BASE_INITIAL),
        }
        with pytest.raises(ValueError, match="dirichlet_A"):
            LearningStrategy().generate_model_code(spec, "no_dirichlet")


class TestOnlineInferenceMode:
    """A1: inference_mode='online' generates the per-timestep filtering loop."""

    def _spec(self) -> dict:
        gnn_file = GNN_FILES / "discrete" / "simple_mdp.md"
        pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
        assert pomdp is not None, f"extraction failed for {gnn_file}"
        return pomdp_to_gnn_spec(pomdp)

    def test_online_option_generates_filtering_script(self, tmp_path: Path) -> None:
        script = tmp_path / "online.jl"
        success, message, _ = render_gnn_to_rxinfer(
            self._spec(), script, options={"inference_mode": "online"}
        )
        assert success, message
        text = script.read_text(encoding="utf-8")
        assert 'const INFERENCE_MODE = "online"' in text
        assert "filtered_posterior" in text
        assert '"inference_mode" => INFERENCE_MODE' in text
        # Action selection still uses the habit prior + EFE
        assert "log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION" in text

    def test_default_stays_batch(self, tmp_path: Path) -> None:
        script = tmp_path / "batch.jl"
        success, message, _ = render_gnn_to_rxinfer(self._spec(), script)
        assert success, message
        text = script.read_text(encoding="utf-8")
        assert "INFERENCE_MODE" not in text
        assert "filtered_posterior" not in text

    def test_spec_declaration_wins_over_option(self, tmp_path: Path) -> None:
        spec = self._spec()
        spec["model_parameters"]["inference_mode"] = "batch"
        script = tmp_path / "declared.jl"
        success, message, _ = render_gnn_to_rxinfer(
            spec, script, options={"inference_mode": "online"}
        )
        assert success, message
        assert "filtered_posterior" not in script.read_text(encoding="utf-8")

    def test_invalid_mode_raises(self) -> None:
        spec = self._spec()
        spec["model_parameters"]["inference_mode"] = "streaming"
        with pytest.raises(ValueError, match="inference_mode"):
            FlatStrategy().generate_model_code(
                build_canonical_pomdp_spec(spec), "simple_mdp"
            )
