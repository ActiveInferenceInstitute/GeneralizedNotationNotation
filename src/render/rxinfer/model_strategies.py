#!/usr/bin/env python3
"""ModelKind-strategy pattern for the RxInfer.jl renderer.

Dispatch model-code generation, animation graph layout, and validation-field
selection to per-``ModelKind`` strategy classes so the renderer no longer
hard-codes the flat POMDP generator.

Each strategy exposes:

- ``generate_model_code(gnn_spec, model_name) -> str`` — the Julia script
  (``@model`` + ``infer()``) for this model kind.
- ``generate_graph_layout(gnn_spec=None) -> dict`` — node positions for
  animations (``{node_name: (x, y)}``).
- ``get_validation_fields() -> list`` — extra validation fields.

``FlatStrategy`` is the canonical flat-POMDP generator.
``MultiAgentStrategy`` renders natively when the spec declares >= 2 complete
agent groups: one genuine ``pomdp_model`` inference per agent (no joint
state-space expansion) coupled through a shared ``env_signal`` affordance
trace (deposit + decay, roadmap MAJ-03). Specs without the per-agent matrix
structure keep the documented joint composition through the flat generator
while stamping their true kind (per-agent recovery happens downstream from
the ``state_factors`` echo).
``HierarchicalStrategy`` renders two-level models natively (slow context
coupled into the fast-state prior) and 3+-level models as the joint
composition. ``FactoredStrategy`` (roadmap D3), ``ContinuousStrategy``
(A2) and ``LearningStrategy`` (D1) render natively against the
``factored_pomdp_model``, ``continuous_pomdp_model`` and
``learning_pomdp_model`` definitions in ``GnnRxInferModels``; each raises
``ValueError`` naming the missing parameterization when a spec reaches it
without the matrices its ``@model`` requires.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from render.pomdp_contract import ModelKind
from render.rxinfer._strategies_continuous import _generate_continuous_code
from render.rxinfer._strategies_factored import _generate_factored_code
from render.rxinfer._strategies_flat import _generate_batch_code, _generate_online_code
from render.rxinfer._strategies_hierarchical import _generate_two_level_code
from render.rxinfer._strategies_learning import _generate_learning_code

__all__ = [
    "ModelStrategy",
    "FlatStrategy",
    "FactoredStrategy",
    "HierarchicalStrategy",
    "MultiAgentStrategy",
    "ContinuousStrategy",
    "LearningStrategy",
    "get_model_strategy",
    "STRATEGY_REGISTRY",
]


def _now() -> str:
    """Return a timestamp string for generated-script headers."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ModelStrategy(ABC):
    """Base class for per-``ModelKind`` RxInfer generation strategies."""

    kind: ModelKind

    # --- hooks ---------------------------------------------------------

    @abstractmethod
    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Return the Julia ``@model`` code for this model kind."""
        ...

    def generate_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return node positions for animations: ``{name: (x, y)}``."""
        return self._default_graph_layout(gnn_spec)

    def get_validation_fields(self) -> List[str]:
        """Return extra validation fields this strategy contributes."""
        return []

    def _default_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Canonical left-to-right POMDP plate layout."""
        return {
            "D": (0.00, 0.85),
            "A": (0.00, 0.55),
            "B": (0.00, 0.25),
            "C": (0.00, -0.05),
            "s": (0.33, 0.85),
            "u": (0.33, 0.25),
            "o": (0.66, 0.55),
            "s'": (0.66, 0.85),
            "G": (1.00, 0.55),
        }


class FlatStrategy(ModelStrategy):
    """Single-factor POMDP (the common case) \u2014 the existing canonical generator."""

    kind = ModelKind.FLAT

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate a genuine RxInfer.jl script with @model + infer() from canonical POMDP matrices.

        Delegates to ``_strategies_flat`` \u2014 see that module for the full
        docstring. The six-phase pipeline and all per-iteration VFE semantics
        are identical to the original monolithic implementation.
        """
        return _generate_batch_code(gnn_spec, model_name, self.kind.value)

    def _generate_online_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the online (filtering) active-inference variant (A1).

        Delegates to ``_strategies_flat`` \u2014 see that module for the full
        docstring.
        """
        return _generate_online_code(gnn_spec, model_name, self.kind.value)

    def generate_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Flat model: single-observation / single-state-factor plate.
        return self._default_graph_layout(gnn_spec)

    def get_validation_fields(self) -> List[str]:
        return [
            "all_beliefs_valid",
            "beliefs_sum_to_one",
            "actions_in_range",
            "inference_converged",
            "vfe_present",
            "belief_entropy_ok",
            "belief_accuracy",
            "belief_accuracy_ok",
        ]


class _JointCompositionStrategy(FlatStrategy):
    """Render via the extractor's joint composition (the flat ``@model``).

    The extractor already composes multiple factors into one joint POMDP
    (C-order ``itertools.product`` over ``state_factors``), and that joint
    model renders and executes through the flat generator. Subclasses stamp
    their true detected kind into the generated script's ``model_kind``
    metadata, and the results JSON echoes ``state_factors`` so downstream
    analysis can recover per-factor marginals from the joint posterior.

    This is a deliberate, documented rendering decision (the
    pre-strategy-pattern behavior), not a silent fallback.
    """


class MultiAgentStrategy(_JointCompositionStrategy):
    """Multiple coordinated agents.

    When the spec declares two or more complete agent groups
    (``A_agentN``/``B_agentN``/``C_agentN``/``D_agentN``) the strategy
    renders natively through the stigmergic generator
    (``_strategies_multiagent``): one genuine ``pomdp_model`` inference per
    agent (no joint state-space expansion) coupled through a shared
    ``env_signal`` affordance trace (deposit + decay). Specs without the
    per-agent matrix structure keep the pre-strategy behavior — the
    extractor's composed joint POMDP through the flat generator — with
    per-agent beliefs recovered downstream from the ``state_factors`` echo
    (roadmap D4). Either way the true kind is stamped.
    """

    kind = ModelKind.MULTI_AGENT

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate native stigmergic code when >= 2 agent groups are declared.

        Delegates to ``_strategies_multiagent`` — see that module for the
        full docstring. Falls back to the composed-joint flat generator
        (documented interim behavior) for specs without per-agent matrices.
        """
        from render.multi_agent_common import has_native_multi_agent_structure

        if has_native_multi_agent_structure(gnn_spec):
            from render.rxinfer._strategies_multiagent import _generate_stigmergic_code

            return _generate_stigmergic_code(gnn_spec, model_name, self.kind.value)
        return super().generate_model_code(gnn_spec, model_name)

    def get_validation_fields(self) -> List[str]:
        return super().get_validation_fields() + [
            "env_signal_trace_valid",
            "per_agent_all_valid",
        ]


class HierarchicalStrategy(FlatStrategy):
    """Two-level slow/fast hierarchical POMDP rendering (roadmap A3).

    Two-level exemplars (``A_level1``/``B_level1``/... plus
    ``A_level2``/``D_level2``) render to the native hierarchical ``@model``
    in ``GnnRxInferModels``: a single Categorical context ``z`` couples into
    the fast-state prior via the column-normalized ``A_level2``, and the
    fast chain is driven by observed actions over ``B_level1``. Inference
    requires the mean-field constraint + uniform marginal initialization
    shipped with the model (Bethe free-energy scoring of the non-square
    coupling hits ReactiveMP's square-matrix ``mul_trace`` assertion
    otherwise \u2014 verified empirically against RxInfer 5.5). Context dynamics
    (``B_level2``) are applied post-hoc as deterministic prior propagation
    and reported as the slow factor's belief trajectory.

    Models declaring 3+ levels render as the extractor's joint composition
    (the pre-strategy-pattern behavior) with ``model_kind`` stamped
    ``hierarchical`` \u2014 a deliberate, documented interim decision recorded in
    the roadmap (native N-level chain rendering is open), not a silent
    fallback.
    """

    kind = ModelKind.HIERARCHICAL

    _REQUIRED_TWO_LEVEL = (
        "A_level1",
        "B_level1",
        "C_level1",
        "A_level2",
        "D_level2",
    )

    @staticmethod
    def _declared_levels(gnn_spec: Dict[str, Any]) -> set:
        matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
        levels = set()
        for key in matrices:
            match = re.match(r"^[ABCDE]_level(\d+)$", str(key))
            if match:
                levels.add(int(match.group(1)))
        return levels

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        levels = self._declared_levels(gnn_spec)
        if levels == {1, 2}:
            matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
            missing = [key for key in self._REQUIRED_TWO_LEVEL if key not in matrices]
            if missing:
                raise ValueError(
                    f"hierarchical model {model_name} is missing per-level "
                    f"matrices required for two-level rendering: {missing}"
                )
            return self._generate_two_level_code(gnn_spec, model_name)
        # 3+ declared levels: joint composition (see class docstring).
        return super().generate_model_code(gnn_spec, model_name)

    def get_validation_fields(self) -> List[str]:
        return super().get_validation_fields() + [
            "context_beliefs_valid",
            "context_beliefs_sum_to_one",
        ]

    def _generate_two_level_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """Generate the native two-level hierarchical RxInfer.jl script.

        Delegates to ``_strategies_hierarchical`` \u2014 see that module for the
        full docstring. The generated Julia script text is identical to the
        original monolithic implementation.
        """
        return _generate_two_level_code(gnn_spec, model_name, self.kind.value)


def _descriptor_name(descriptors: Any, index: int, default: str) -> str:
    """Return ``descriptors[index]["name"]`` from a GNN descriptor list."""
    if isinstance(descriptors, list) and len(descriptors) > index:
        entry = descriptors[index]
        if isinstance(entry, dict) and entry.get("name"):
            return str(entry["name"])
    return default


class FactoredStrategy(ModelStrategy):
    """Two-factor / two-modality POMDP rendered natively (roadmap D3).

    The exemplar's ``## Equations`` declare the mean-field factorization
    ``Q(s_f0, s_f1) = Q(s_f0) Q(s_f1)``, and ``factored_constraints()``
    states exactly that cut \u2014 so the posterior family IS the declared
    model, not an approximation bolted on afterwards. Factor 0 is the
    action-driven (controllable) chain over ``B_f0``; factor 1 is the
    passive chain over the static ``B_f1``. Modality 0 depends on BOTH
    factors through the 3-tensor ``A_m0``; modality 1 depends on factor 0
    alone through ``A_m1``.

    The native path REQUIRES the per-factor matrices. When they are absent
    this raises ``ValueError`` rather than dropping to the extractor's
    joint composition: a model detected FACTORED without per-factor
    matrices is a contract violation, not a case for a quieter render.
    """

    kind = ModelKind.FACTORED

    _REQUIRED_MATRICES = ("A_m0", "A_m1", "B_f0", "B_f1", "D_f0", "D_f1")

    def get_validation_fields(self) -> List[str]:
        return [
            "all_beliefs_valid",
            "beliefs_sum_to_one",
            "actions_in_range",
            "inference_converged",
            "vfe_present",
            "belief_entropy_ok",
            "belief_accuracy",
            "belief_accuracy_ok",
            "factor1_beliefs_valid",
            "factor1_beliefs_sum_to_one",
        ]

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the native two-factor mean-field RxInfer.jl script.

        Delegates to ``_strategies_factored`` \u2014 see that module for the full
        docstring. The generated Julia script text is identical to the original
        monolithic implementation.
        """
        return _generate_factored_code(gnn_spec, model_name, self.kind.value)


class ContinuousStrategy(ModelStrategy):
    """Linear-Gaussian state-space rendering for continuous specs (A2).

    Continuous exemplars carry an authored continuous parameterization
    (``F``/``H``/``Q``/``R`` plus ``prior_mean``/``prior_cov``) alongside
    their discretized POMDP stand-in. This strategy renders those keys onto
    ``continuous_pomdp_model`` \u2014 the LGSSM ``@model`` in
    ``GnnRxInferModels`` \u2014 which needs neither constraints nor
    initialization (it is fully conjugate; belief propagation converges in
    one sweep).

    Deriving F/H/Q/R from discrete A/B/C/D would fabricate data, so a spec
    reaching this strategy without them raises ``ValueError`` naming the
    missing keys.
    """

    kind = ModelKind.CONTINUOUS

    _REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")

    def get_validation_fields(self) -> List[str]:
        return [
            "vfe_finite",
            "means_finite",
            "posterior_cov_psd",
            "inference_converged",
            "rmse_vs_true",
            "rmse_finite",
        ]

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the native linear-Gaussian RxInfer.jl script.

        Delegates to ``_strategies_continuous`` \u2014 see that module for the full
        docstring. The generated Julia script text is identical to the original
        monolithic implementation.
        """
        return _generate_continuous_code(gnn_spec, model_name, self.kind.value)


class LearningStrategy(ModelStrategy):
    """Dirichlet likelihood learning rendered natively (roadmap D1).

    Specs carrying ``dirichlet_A`` pseudo-counts render onto
    ``learning_pomdp_model``, where the likelihood ``A`` is a latent
    ``DirichletCollection`` rather than a fixed constant. The structured
    mean-field cut ``q(s, A) = q(s)q(A)`` plus marginal initialization of
    both sides is required by RxInfer 5.5, and the counts must break
    column-permutation symmetry (a uniform prior converges to a
    label-switched optimum that the free energy alone will not catch \u2014
    which is why ``a_distance_posterior`` is a hard gate).
    """

    kind = ModelKind.LEARNING

    _REQUIRED_KEYS = ("dirichlet_A", "A", "B", "C", "D")

    def get_validation_fields(self) -> List[str]:
        return [
            "all_beliefs_valid",
            "beliefs_sum_to_one",
            "actions_in_range",
            "inference_converged",
            "vfe_present",
            "belief_entropy_ok",
            "belief_accuracy",
            "belief_accuracy_ok",
            "a_learning_improved",
            "a_posterior_columns_normalized",
        ]

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the native Dirichlet-likelihood-learning RxInfer.jl script.

        Delegates to ``_strategies_learning`` \u2014 see that module for the full
        docstring. The generated Julia script text is identical to the original
        monolithic implementation.
        """
        return _generate_learning_code(gnn_spec, model_name, self.kind.value)


STRATEGY_REGISTRY: Dict[ModelKind, ModelStrategy] = {
    ModelKind.FLAT: FlatStrategy(),
    ModelKind.FACTORED: FactoredStrategy(),
    ModelKind.HIERARCHICAL: HierarchicalStrategy(),
    ModelKind.MULTI_AGENT: MultiAgentStrategy(),
    ModelKind.CONTINUOUS: ContinuousStrategy(),
    ModelKind.LEARNING: LearningStrategy(),
}


def get_model_strategy(model_kind: ModelKind) -> ModelStrategy:
    """Return the registered strategy for a ``ModelKind``."""
    return STRATEGY_REGISTRY[model_kind]
