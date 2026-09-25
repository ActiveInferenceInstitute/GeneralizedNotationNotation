"""Static per-backend complexity-bound registry for GNN models.

Pure data + applicability predicates consumed by
:mod:`gnn.analysis.complexity.estimator`. Stdlib only — no framework or
executor imports, so the registry is importable without the execute stack.

Every bound is an upper-bound ARGUMENT over declared structure, labeled
``[ESTIMATE]`` in its asymptotic string; the numeric drivers that
parametrize the formula travel in the receipt row's ``drivers`` dict.
Bounds are never measurements and never fabricate a total where the
planning horizon is not numeric: when the horizon is missing, ``"Unbounded"``,
or symbolic, bounds degrade to their per-step form and say so.

Backend keys and order are pinned by the wave-8 brief: the executor
registry's 10 keys (pymdp, rxinfer, discopy, activeinference_jl, jax,
numpyro, pytorch, ngclearn, lean, stan) plus bnlearn, the render-only
registry-gated backend (``render/framework_registry.py`` bnlearn entry).
``jax`` carries two family variants — kronecker-factorized for discrete
kinds, dense linear-Gaussian for continuous kinds; the estimator resolves
exactly one row per framework from this registry, in ``BACKEND_ORDER``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

BACKEND_ORDER: tuple[str, ...] = (
    "pymdp",
    "rxinfer",
    "discopy",
    "activeinference_jl",
    "jax",
    "numpyro",
    "pytorch",
    "ngclearn",
    "lean",
    "stan",
    "bnlearn",
)

#: Model kinds that exercise discrete (categorical) machinery.
DISCRETE_KINDS: frozenset[str] = frozenset(
    {"flat", "factored", "hierarchical", "multi_agent", "nonstationary", "learning"}
)

#: Continuous family marker emitted by ``detect_model_kinds``.
CONTINUOUS_KIND: str = "continuous"

#: The discrete+continuous family mix no backend renders whole.
HYBRID_KIND: str = "hybrid"


@dataclass(frozen=True)
class StructureDims:
    """Declared-structure view a bound is parameterized by.

    ``horizon`` is an int when the spec declares a numeric planning
    horizon; the string ``"Unbounded"`` when absent/unbounded; or a
    symbolic string (e.g. ``"T"``). The latter two mean *no numeric
    total bound*: bounds degrade to their per-step form and the drivers
    carry the horizon label verbatim.
    """

    horizon: int | str
    has_numeric_horizon: bool
    joint_state_dim: int
    state_space_dim_total: int
    max_variable_dim: int
    max_factor_arity: int
    edge_count: int
    variable_count: int
    agents: int
    regimes: int


ApplicableFn = Callable[[frozenset[str], StructureDims], bool]
AsymptoticFn = Callable[[StructureDims], str]


@dataclass(frozen=True)
class BackendBound:
    """One (framework, family) bound entry."""

    framework: str
    family: str
    complexity_class: str
    notes: str
    applicable: ApplicableFn
    asymptotic: AsymptoticFn


def _discrete_applicable(kinds: frozenset[str], dims: StructureDims) -> bool:
    """Discrete message passing applies unless a continuous/hybrid mix is declared."""
    return bool(kinds & DISCRETE_KINDS) and not (kinds & {CONTINUOUS_KIND, HYBRID_KIND})


def _continuous_applicable(kinds: frozenset[str], dims: StructureDims) -> bool:
    """Dense linear-Gaussian bounds apply only to the continuous family."""
    return CONTINUOUS_KIND in kinds


def _always_applicable(kinds: frozenset[str], dims: StructureDims) -> bool:
    """Family applies to any declared structure."""
    return True


def _unbounded_note(dims: StructureDims) -> str:
    """Suffix documenting why no numeric total bound is emitted."""
    if dims.has_numeric_horizon:
        return ""
    return " (horizon Unbounded: no numeric total bound)"


def _factorized_asymptotic(dims: StructureDims) -> str:
    """Discrete per-timestep message passing over the factor product."""
    if dims.has_numeric_horizon:
        return "O(T * prod_f |s_f| * prod_m |o_m| * |a|) [ESTIMATE]"
    return (
        "O(prod_f |s_f| * prod_m |o_m| * |a|) per timestep [ESTIMATE]"
        + _unbounded_note(dims)
    )


def _kronecker_asymptotic(dims: StructureDims) -> str:
    """Kronecker-factorized JAX ops: per-factor contractions, no joint materialization."""
    if dims.has_numeric_horizon:
        return "O(T * sum_f |s_f| * |o| * |a|) [ESTIMATE] (kronecker-factorized: no joint state materialization)"
    return (
        "O(sum_f |s_f| * |o| * |a|) per timestep [ESTIMATE]"
        " (kronecker-factorized: no joint state materialization)"
        + _unbounded_note(dims)
    )


def _lgssm_asymptotic(dims: StructureDims) -> str:
    """Dense linear-Gaussian (Kalman-like) updates over the joint state."""
    if dims.has_numeric_horizon:
        return "O(T * d^3) time, O(d^2) memory [ESTIMATE]"
    return "O(d^3) time per step, O(d^2) memory [ESTIMATE]" + _unbounded_note(dims)


def _sampling_asymptotic(dims: StructureDims) -> str:
    """Sampling-family bound, reported per sample; samples is a runner knob."""
    return (
        "O(per_sample_cost) [ESTIMATE]; total = O(samples * per_sample_cost)"
        " — samples is a runner knob; bound reported per-sample"
    )


def _structure_learning_asymptotic(dims: StructureDims) -> str:
    """Score-based discrete-network structure search."""
    return "O(2^N_v * score_cost) worst-case score-based structure search [ESTIMATE]"


def _composition_asymptotic(dims: StructureDims) -> str:
    """Categorical string-diagram composition size."""
    return "O(sum_f |s_f| + E) diagram composition size [ESTIMATE]"


def _verification_asymptotic(dims: StructureDims) -> str:
    """Verification backend: class-only label, never a numeric bound."""
    return "class-only: proof cost is not numerically estimated"


BACKEND_BOUNDS: tuple[BackendBound, ...] = (
    BackendBound(
        framework="pymdp",
        family="exact-factorized",
        complexity_class="polynomial",
        notes=(
            "Per-timestep discrete message passing over the factor product"
            " prod_f |s_f| x observation space x action space, times the"
            " planning horizon. Continuous and hybrid specs are refused"
            " (discrete A/B/C/D machinery only)."
        ),
        applicable=_discrete_applicable,
        asymptotic=_factorized_asymptotic,
    ),
    BackendBound(
        framework="rxinfer",
        family="exact-dense-LGSSM",
        complexity_class="polynomial",
        notes=(
            "Dense linear-Gaussian message passing over the joint continuous"
            " state d: Kalman-like per-step updates cost O(d^3) time and"
            " O(d^2) memory. Continuous family only."
        ),
        applicable=_continuous_applicable,
        asymptotic=_lgssm_asymptotic,
    ),
    BackendBound(
        framework="discopy",
        family="categorical-composition",
        complexity_class="polynomial",
        notes=(
            "Categorical string-diagram composition; diagram size tracks the"
            " sum of factor state dims plus declared edges. Continuous specs"
            " are unsupported (render framework_registry discopy entry)."
        ),
        applicable=_discrete_applicable,
        asymptotic=_composition_asymptotic,
    ),
    BackendBound(
        framework="activeinference_jl",
        family="exact-factorized",
        complexity_class="polynomial",
        notes=(
            "Julia discrete POMDP message passing; same factor-product shape"
            " as pymdp. Registry supports_continuous=False; Julia process"
            " startup is excluded from the bound."
        ),
        applicable=_discrete_applicable,
        asymptotic=_factorized_asymptotic,
    ),
    BackendBound(
        framework="jax",
        family="exact-factorized",
        complexity_class="polynomial",
        notes=(
            "Kronecker-factorized JAX ops: per-factor contractions sum over"
            " factors instead of materializing the joint state. Selected for"
            " discrete kinds when the spec declares no continuous family."
        ),
        applicable=_discrete_applicable,
        asymptotic=_kronecker_asymptotic,
    ),
    BackendBound(
        framework="jax",
        family="exact-dense-LGSSM",
        complexity_class="polynomial",
        notes=(
            "Dense linear-Gaussian updates in JAX (scan over the horizon):"
            " O(d^3) time and O(d^2) memory per step. Selected when the spec"
            " declares the continuous family."
        ),
        applicable=_continuous_applicable,
        asymptotic=_lgssm_asymptotic,
    ),
    BackendBound(
        framework="numpyro",
        family="sampling",
        complexity_class="sampling",
        notes=(
            "MCMC/sampling inference: the model-independent bound is per"
            " sample; the sample count is an execution knob (Step 12), not a"
            " model property."
        ),
        applicable=_always_applicable,
        asymptotic=_sampling_asymptotic,
    ),
    BackendBound(
        framework="pytorch",
        family="sampling",
        complexity_class="sampling",
        notes=(
            "Autograd-driven tensor simulation: per-sample trajectory cost;"
            " the sample count is a runner knob."
        ),
        applicable=_always_applicable,
        asymptotic=_sampling_asymptotic,
    ),
    BackendBound(
        framework="ngclearn",
        family="sampling",
        complexity_class="sampling",
        notes=(
            "Predictive-processing neural simulation: per-sample, per-step"
            " updates; the sample count is a runner knob."
        ),
        applicable=_always_applicable,
        asymptotic=_sampling_asymptotic,
    ),
    BackendBound(
        framework="lean",
        family="verification",
        complexity_class="verification",
        notes=(
            "Verification backend: proof cost only; the complexity label is"
            " class-only and never carries a numeric bound."
        ),
        applicable=_always_applicable,
        asymptotic=_verification_asymptotic,
    ),
    BackendBound(
        framework="stan",
        family="sampling",
        complexity_class="sampling",
        notes=(
            "HMC/NUTS sampling: per-sample cost; warmup and chain count are"
            " runner knobs."
        ),
        applicable=_always_applicable,
        asymptotic=_sampling_asymptotic,
    ),
    BackendBound(
        framework="bnlearn",
        family="structure-learning",
        complexity_class="combinatorial-search",
        notes=(
            "Render-only structure learning (registry-gated,"
            " render/framework_registry.py bnlearn entry): score-based"
            " discrete-network search; executed only when the bnlearn extra"
            " is installed."
        ),
        applicable=_always_applicable,
        asymptotic=_structure_learning_asymptotic,
    ),
)
