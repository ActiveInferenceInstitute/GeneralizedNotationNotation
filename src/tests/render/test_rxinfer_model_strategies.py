"""Regression tests for RxInfer ModelKind detection + strategy dispatch.

Pins two invariants the 2026-08-05 red-team review found broken:

1. ``detect_model_kind`` is STRUCTURAL — prose in a ModelName or annotation
   must never change how a model renders (the old ``str(gnn_spec)``
   substring scan misrouted ``temporal_hierarchy.md`` on the word
   "Hierarchy" in its name and made every exemplar one doc-comment away
   from a render failure).
2. Every GNN exemplar renders through the real pipeline path (the
   "45/45 render" contract), with the intended kind taxonomy.

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
    FactoredStrategy,
    FlatStrategy,
    HierarchicalStrategy,
    MultiAgentStrategy,
    get_model_strategy,
)
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GNN_FILES = PROJECT_ROOT / "input" / "gnn_files"

# The intended kind for every non-flat exemplar; everything else is FLAT.
EXPECTED_NON_FLAT = {
    "hierarchical/hierarchical_pomdp.md": ModelKind.HIERARCHICAL,
    "hierarchical/temporal_hierarchy.md": ModelKind.HIERARCHICAL,
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
    assert len(files) == 45, f"expected 45 exemplars, found {len(files)}"
    return files


def _canonical_spec(gnn_file: Path) -> dict:
    pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
    assert pomdp is not None, f"extraction failed for {gnn_file}"
    return build_canonical_pomdp_spec(pomdp_to_gnn_spec(pomdp))


class TestExemplarKindTaxonomy:
    """Every exemplar detects its intended kind through the real path."""

    def test_all_45_exemplars_detect_expected_kind(self) -> None:
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

    def test_all_45_exemplars_render(self, tmp_path: Path) -> None:
        """The 45/45 render contract, through the public renderer entry."""
        failures = []
        for gnn_file in _exemplar_files():
            pomdp = extract_pomdp_from_file(gnn_file, strict_validation=True)
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
        assert isinstance(get_model_strategy(ModelKind.FACTORED), FactoredStrategy)
        assert isinstance(get_model_strategy(ModelKind.MULTI_AGENT), MultiAgentStrategy)
        assert isinstance(FactoredStrategy(), FlatStrategy)
        assert isinstance(MultiAgentStrategy(), FlatStrategy)

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
