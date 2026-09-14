"""Structural wrapper specs classify as STRUCTURAL, not discrete POMDPs.

Regression coverage for issue #111: a structural Markov-blanket wrapper spec
(e.g. the NEST programme's D02/D03 deliverables) declares boundary structure —
sensory/active/internal partitions with illustrative dimensions — but ships no
generative parameterization (no discrete ``A/B/C/D[/E]``, no continuous
``F/H/Q/R``). v3.2.0's classifier used to fall through to FLAT, and Step 11
then failed the render with the cryptic
``Missing required matrices: ['A', 'B', 'C', 'D']``.

The fixed contract:

1. ``detect_model_kind`` returns the third classification
   ``ModelKind.STRUCTURAL`` for such specs (discrete and continuous kinds are
   unchanged).
2. The Step 11 pipeline reports structural wrappers as ``unsupported``
   (render-only / informational) — the same accounting continuous models get
   from categorical backends — instead of failing the framework render.
3. The public ``render_gnn_spec`` entry rejects a wrapper with the clear
   ``structural-spec: no renderable form`` message rather than a missing-
   matrices error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from gnn.extract.pomdp_extractor import extract_pomdp_from_file
from gnn.render.pomdp_contract import (
    ModelKind,
    detect_model_kind,
    detect_pomdp_space_model_kind,
)
from gnn.render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from gnn.render.processor import render_gnn_spec
from gnn.render.rxinfer.model_strategies import get_model_strategy

STRUCTURAL_WRAPPER_MD = """\
# GNN Example: Wrapper Blanket
# GNN Version: 1.0

## GNNSection
StructuralBlanket

## GNNVersionAndFlags
GNN v1

## ModelName
Wrapper Blanket Pattern

## ModelAnnotation
A structural Markov-blanket wrapper: declares what crosses the model boundary
without shipping a generative parameterization. Declares structure, not values.

## StateSpaceBlock
# Sensory (inbound) partition
d[2,12,type=float]       # sensory blanket channel
cap[1,type=float]        # sensory capacity scalar
# Active (outbound) partition
p[2,12,type=float]       # active blanket channel
e[1,type=float]          # active effort scalar
# Internal partition
K[4,type=float]          # internal stock
x[4,12,type=float]       # internal trajectory

## Connections
d>s
s>p
cap>s
s>e
x>s
s>K

## InitialParameterization
# Declares structure, not values.
d={(0.0,0.0)}
cap={(200000.0)}
p={(0.0,0.0)}
e={(1.0)}
K={(0.0,0.0,0.0,0.0)}
x={(0.0,0.0,0.0,0.0)}

## ModelParameters
num_timesteps: 12

## Footer
Wrapper blanket v1.0 - structural, non-generative.

## Signature
Cryptographic signature goes here
"""


def _write_wrapper(tmp_path: Path) -> Path:
    path = tmp_path / "wrapper_blanket.gnn.md"
    path.write_text(STRUCTURAL_WRAPPER_MD, encoding="utf-8")
    return path


class TestStructuralClassification:
    """Structural wrapper specs get the third classification — not discrete."""

    def test_wrapper_spec_is_structural_not_flat(self) -> None:
        """A blanket parameterization (non-contract keys only) is STRUCTURAL."""
        spec: Dict[str, Any] = {
            "model_name": "Wrapper Blanket",
            "gnn_section": "StructuralBlanket",
            "model_parameters": {"num_timesteps": 12},
            "initialparameterization": {
                "d": [[0.0, 0.0]],
                "cap": [200000.0],
                "p": [[0.0, 0.0]],
                "e": [1.0],
            },
        }
        assert detect_model_kind(spec) is ModelKind.STRUCTURAL

    def test_extracted_wrapper_space_is_structural(self, tmp_path: Path) -> None:
        """The extracted space of a wrapper file classifies STRUCTURAL.

        The extractor labels it with the coarse ``model_kind == "discrete"``
        (the exact condition behind issue #111); the render-side classifier
        must refine that to STRUCTURAL.
        """
        pomdp = extract_pomdp_from_file(_write_wrapper(tmp_path), strict_validation=True)
        assert pomdp is not None
        assert detect_pomdp_space_model_kind(pomdp) is ModelKind.STRUCTURAL

    def test_structural_spec_view_round_trips(self, tmp_path: Path) -> None:
        """pomdp_to_gnn_spec emits a stamped structural view, no canonical A/B/C/D."""
        pomdp = extract_pomdp_from_file(_write_wrapper(tmp_path), strict_validation=True)
        assert pomdp is not None
        spec = pomdp_to_gnn_spec(pomdp)
        assert spec["model_kind"] == "structural"
        assert "A" not in spec["initialparameterization"]
        assert "B" not in spec["initialparameterization"]
        assert detect_model_kind(spec) is ModelKind.STRUCTURAL

    def test_explicit_structural_stamp_is_honored(self) -> None:
        """``model_kind == "structural"`` wins, mirroring the continuous stamp."""
        spec: Dict[str, Any] = {
            "model_kind": "structural",
            "model_name": "Blanket",
            "initialparameterization": {"d": [[0.0, 0.0]]},
        }
        assert detect_model_kind(spec) is ModelKind.STRUCTURAL

    def test_partial_continuous_params_are_structural(self) -> None:
        """F/H without the full F/H/Q/R contract is structure, not continuous."""
        spec: Dict[str, Any] = {
            "model_name": "Partial",
            "initialparameterization": {"F": [[1.0]], "H": [[1.0]]},
        }
        assert detect_model_kind(spec) is ModelKind.STRUCTURAL

    def test_single_discrete_contract_key_is_not_structural(self) -> None:
        """One A–E contract key keeps the discrete (FLAT) classification."""
        spec: Dict[str, Any] = {
            "model_name": "PriorOnly",
            "initialparameterization": {"D": [0.5, 0.5]},
        }
        assert detect_model_kind(spec) is ModelKind.FLAT


class TestKindRegressions:
    """Discrete and continuous classifications are unchanged by STRUCTURAL."""

    def test_flat_discrete_spec_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "model_name": "Test",
            "model_parameters": {"num_hidden_states": 4, "num_obs": 4},
            "initialparameterization": {
                "A": [[1, 0], [0, 1]],
                "B": [[1, 0], [0, 1]],
                "C": [0, 1],
                "D": [0.5, 0.5],
            },
        }
        assert detect_model_kind(spec) is ModelKind.FLAT

    def test_factored_spec_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "model_name": "Factored",
            "model_parameters": {"num_factors": 2},
            "initialparameterization": {
                "A": [[1, 0], [0, 1]],
                "B": [[1, 0], [0, 1]],
                "C": [0, 1],
                "D": [0.5, 0.5],
            },
        }
        assert detect_model_kind(spec) is ModelKind.FACTORED

    def test_hierarchical_spec_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "gnn_section": "hierarchical",
            "initialparameterization": {"A": [[1, 0], [0, 1]]},
        }
        assert detect_model_kind(spec) is ModelKind.HIERARCHICAL

    def test_multi_agent_spec_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "initialparameterization": {"nr_agents": 2, "A_agent1": [[1.0, 0.0]]},
        }
        assert detect_model_kind(spec) is ModelKind.MULTI_AGENT

    def test_learning_spec_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "initialparameterization": {"dirichlet_A": [1.0, 1.0]},
        }
        assert detect_model_kind(spec) is ModelKind.LEARNING

    def test_continuous_section_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "gnn_section": "continuous",
            "initialparameterization": {},
        }
        assert detect_model_kind(spec) is ModelKind.CONTINUOUS

    def test_continuous_params_unchanged(self) -> None:
        spec: Dict[str, Any] = {
            "model_name": "LGSSM",
            "initialparameterization": {
                "F": [[0.5]],
                "H": [[1.0]],
                "Q": [[0.1]],
                "R": [[0.2]],
                "prior_mean": [0.0],
                "prior_cov": [[1.0]],
            },
        }
        assert detect_model_kind(spec) is ModelKind.CONTINUOUS


class TestStep11StructuralRenderPath:
    """Step 11 accepts a structural wrapper without the cryptic failure."""

    def test_pipeline_reports_unsupported_not_failed(self, tmp_path: Path) -> None:
        """Every framework is ``unsupported`` — none failed, none attempted."""
        pomdp = extract_pomdp_from_file(_write_wrapper(tmp_path), strict_validation=True)
        assert pomdp is not None
        processor = POMDPRenderProcessor(tmp_path / "render_out")
        result = processor.process_pomdp_for_all_frameworks(pomdp)
        frameworks = sorted(result["framework_results"])
        assert frameworks, "expected at least one registered framework"
        for framework, framework_result in result["framework_results"].items():
            assert framework_result["unsupported"] is True, framework
            assert framework_result["status"] == "unsupported", framework
            assert framework_result["success"] is False, framework
            assert framework_result["output_files"] == [], framework
            assert "structural-spec" in framework_result["message"], framework
            assert "Missing required matrices" not in framework_result["message"]
        # v3.2.0 accounting: unsupported frameworks are excluded from the
        # success denominator — the run is neither failed nor attempted.
        assert result["overall_success"] is True
        summary_file = Path(result["summary_file"])
        summary = summary_file.read_text(encoding="utf-8")
        assert "frameworks_unsupported" in summary
        assert "Missing required matrices" not in summary

    def test_render_gnn_spec_rejects_wrapper_with_clear_message(
        self, tmp_path: Path
    ) -> None:
        """The public per-target entry says structural-spec, never cryptic."""
        pomdp = extract_pomdp_from_file(_write_wrapper(tmp_path), strict_validation=True)
        assert pomdp is not None
        spec = pomdp_to_gnn_spec(pomdp)
        success, message, output_files = render_gnn_spec(
            spec, "rxinfer", tmp_path / "single"
        )
        assert success is False
        assert output_files == []
        assert "structural-spec" in message
        assert "Missing required matrices" not in message
        assert "not compatible" not in message

    def test_cli_parse_summary_rehydrates_wrapper_to_clear_message(
        self, tmp_path: Path
    ) -> None:
        """The CLI rehydrate path (parse summary → extractor → spec view)
        yields the structural-spec message for a wrapper, never the cryptic
        canonicalization errors it used to raise en route."""
        summary: Dict[str, Any] = {
            "success": True,
            "file_path": str(_write_wrapper(tmp_path)),
            "sections": ["GNNSection", "StateSpaceBlock", "InitialParameterization"],
            "variables": [],
            "connections": [],
            "model_name": "Wrapper Blanket Pattern",
        }
        success, message, output_files = render_gnn_spec(
            summary, "rxinfer", tmp_path / "cli_out"
        )
        assert success is False
        assert output_files == []
        assert "structural-spec" in message
        assert "Missing required matrices" not in message
        assert "Factored POMDP" not in message

    def test_strategy_dispatch_rejects_structural_loudly(self) -> None:
        """STRUCTURAL has no render strategy — a clear error, not a KeyError."""
        with pytest.raises(ValueError, match="structural-spec"):
            get_model_strategy(ModelKind.STRUCTURAL)

    def test_graph_backed_target_renders_wrapper_structure(
        self, tmp_path: Path
    ) -> None:
        """Graph-backed targets (stan) render declared structure — not gated.

        The structural gate only covers targets that canonicalise to a
        discrete parameterization; stan/bnlearn/discopy legitimately render a
        parameterization-free spec from its variables and connections.
        """
        pomdp = extract_pomdp_from_file(_write_wrapper(tmp_path), strict_validation=True)
        assert pomdp is not None
        spec = pomdp_to_gnn_spec(pomdp)
        success, message, files = render_gnn_spec(spec, "stan", tmp_path / "stan_out")
        assert success, message
        assert files and Path(files[0]).exists()
