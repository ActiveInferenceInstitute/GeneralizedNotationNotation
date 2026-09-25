"""Static complexity estimator: parsed GNN model -> deterministic receipt.

Accepts a :class:`~gnn.parsers.common.GNNInternalRepresentation` or a
filesystem path and emits the pinned ``gnn.complexity_estimate/v1`` receipt
(the wave-8 cross-lane contract). Pure stdlib plus existing GNN parser /
type-checker / render-contract modules; zero framework and zero executor
imports, so the estimator is importable without the execute stack.

Determinism: model kinds are sorted, per-backend rows follow the fixed
``bounds.BACKEND_ORDER``, and :func:`to_json_text` emits sorted-key JSON.
Bounds are ESTIMATE-labeled arguments over declared structure (see
:mod:`gnn.analysis.complexity.bounds`); a non-numeric planning horizon never
produces a numeric total bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from gnn.parsers import parse_gnn_file_structured
from gnn.parsers.common import DataType, GNNInternalRepresentation
from gnn.render.pomdp_contract import detect_model_kinds
from gnn.type_checker import GNNTypeChecker

from . import bounds

RECEIPT_TYPE = "gnn.complexity_estimate/v1"
ESTIMATOR_VERSION = "1"
UNBOUNDED_HORIZON = "Unbounded"

#: Declared data types counted as discrete / continuous variables.
DISCRETE_DATA_TYPES = frozenset(
    {DataType.CATEGORICAL, DataType.BINARY, DataType.INTEGER}
)
CONTINUOUS_DATA_TYPES = frozenset(
    {DataType.CONTINUOUS, DataType.FLOAT, DataType.COMPLEX}
)

#: ``B_regime``-style tensor variables carry the regime count as first dim.
_REGIME_VARIABLE = re.compile(r"^B_regime\d*$", re.IGNORECASE)

#: Square linear-Gaussian system-matrix declarations (``[d, d]``).
_LGSSM_SYSTEM_MATRIX_NAMES = frozenset({"F", "H", "Q", "R"})

#: Per-agent matrix declarations (``A_agent2``) — the agent evidence
#: ``detect_model_kinds`` also accepts when ``nr_agents`` is undeclared.
_AGENT_MATRIX_KEY = re.compile(r"^[ABCDE]_agent(\d+)$", re.IGNORECASE)


def estimate_model_complexity(
    model_or_path: GNNInternalRepresentation | str | Path,
) -> dict[str, Any]:
    """Estimate static per-backend complexity for a parsed GNN model.

    Args:
        model_or_path: A parsed ``GNNInternalRepresentation`` or a path to a
            GNN spec file.

    Returns:
        The pinned ``gnn.complexity_estimate/v1`` receipt dict.

    Raises:
        FileNotFoundError: Path form with a missing file (propagated).
        ValueError: Path form whose spec does not parse to a model.
        TypeError: Any other input type.
    """
    if isinstance(model_or_path, (str, Path)):
        path = Path(model_or_path)
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        parsed = parse_gnn_file_structured(path)
        model = parsed.model
        if model is None:
            raise ValueError(f"failed to parse {path}: {list(parsed.errors)}")
        sha256 = hashlib.sha256(raw).hexdigest()
        source_path = str(path)
    elif isinstance(model_or_path, GNNInternalRepresentation):
        model = model_or_path
        text = _content_from_raw_sections(model)
        canonical = json.dumps(
            model.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        source_path = ""
    else:
        raise TypeError(
            "estimate_model_complexity accepts a GNNInternalRepresentation or"
            f" a filesystem path, got {type(model_or_path).__name__}"
        )

    structure = _structure_stats(model, text)
    spec = model.to_dict()
    kinds = sorted(kind.value for kind in detect_model_kinds(spec))
    dims = _structure_dims(structure, spec, model, _validate_content(text))
    return {
        "receipt_type": RECEIPT_TYPE,
        "model": {
            "name": model.model_name,
            "source_sha256": sha256,
            "path": source_path,
        },
        "structure": structure,
        "model_kinds": kinds,
        "per_backend": _per_backend_rows(set(kinds), dims),
        "estimator_version": ESTIMATOR_VERSION,
    }


def to_json_text(receipt: dict[str, Any]) -> str:
    """Serialize a receipt to stable JSON: sorted keys, fixed indent."""
    return json.dumps(receipt, sort_keys=True, indent=2)


def _content_from_raw_sections(model: GNNInternalRepresentation) -> str:
    """Reconstruct raw spec text from the parser's raw section map.

    The markdown parser stores every ``## <name>`` section body verbatim;
    ``gnn.validation.structure`` reassembles content the same way. Empty
    when the model object was built without raw sections — the estimator
    then degrades to the parse-object dims (symbolic tokens already
    collapse to 1 in the generic parser tier).
    """
    if not model.raw_sections:
        return ""
    return "\n\n".join(
        f"## {name}\n{body}" for name, body in model.raw_sections.items()
    )


def _validate_content(text: str) -> dict[str, Any] | None:
    """Run the type_checker content layer (symbolic dims + resource estimation).

    Returns ``None`` when no raw content is available; callers fall back to
    the parse-object dims.
    """
    if not text:
        return None
    return GNNTypeChecker().validate_content(text)


def _structure_stats(
    model: GNNInternalRepresentation, text: str
) -> dict[str, Any]:
    """Structural statistics from the parse object + type-checker layer."""
    connections = model.connections
    variables = model.variables
    discrete_var_count = sum(
        1 for variable in variables if variable.data_type in DISCRETE_DATA_TYPES
    )
    continuous_var_count = sum(
        1 for variable in variables if variable.data_type in CONTINUOUS_DATA_TYPES
    )
    validation = _validate_content(text)
    total_state_space_dim, max_variable_dim = _resolved_dims(validation, variables)
    time_specification = model.time_specification
    if validation is not None:
        is_dynamic = bool(validation["time_dynamics"]["is_dynamic"])
    else:
        is_dynamic = bool(
            time_specification and time_specification.time_type.lower() == "dynamic"
        )
    return {
        "variable_count": len(variables),
        "edge_count": len(connections),
        "factor_arities": [
            len(conn.source_variables) + len(conn.target_variables)
            for conn in connections
        ],
        "total_state_space_dim": total_state_space_dim,
        "max_variable_dim": max_variable_dim,
        "discrete_var_count": discrete_var_count,
        "continuous_var_count": continuous_var_count,
        "time": {
            "time_type": time_specification.time_type if time_specification else "Static",
            "discretization": (time_specification.discretization or "")
            if time_specification
            else "",
            "horizon": _normalize_horizon(
                time_specification.horizon if time_specification else None
            ),
            "is_dynamic": is_dynamic,
        },
    }


def _resolved_dims(
    validation: dict[str, Any] | None,
    variables: list[Any],
) -> tuple[int, int]:
    """Total and max per-variable state-space sizes, symbolic dims resolved.

    Primary source: the type_checker layer's ``variables`` metadata, whose
    ``total_elements`` is computed from the *resolved* form of the symbolic
    StateSpaceBlock declarations (variable-backed references like ``G[pi]``
    resolve against the declared ``pi``; unresolvable symbols degrade to 1
    with a diagnostic). Fallback: the parse object's integer dims.
    """
    if validation is not None:
        totals = [
            int(entry.get("total_elements", 1) or 1)
            for entry in validation["variables"]
        ]
        if totals:
            return sum(totals), max(totals)
    totals = [
        math.prod(variable.dimensions) if variable.dimensions else 1
        for variable in variables
    ]
    return sum(totals), max(totals, default=0)


def _normalize_horizon(horizon: int | str | None) -> int | str:
    """Receipt horizon: an int, or the ``"Unbounded"`` class-only label.

    ``None`` and any unbounded marker normalize to ``"Unbounded"``; numeric
    strings are coerced; a symbolic horizon string (e.g. ``"T"``) is kept
    verbatim and bounds treat it as non-numeric — per-step bound only,
    never a fabricated total.
    """
    if horizon is None:
        return UNBOUNDED_HORIZON
    if isinstance(horizon, int):
        return horizon
    text = str(horizon).strip()
    if "unbound" in text.lower():
        return UNBOUNDED_HORIZON
    try:
        return int(text)
    except ValueError:
        return text


def _structure_dims(
    structure: dict[str, Any],
    spec: dict[str, Any],
    model: GNNInternalRepresentation,
    validation: dict[str, Any] | None,
) -> bounds.StructureDims:
    """Pack the receipt structure into the registry's bound-parameter view."""
    horizon = structure["time"]["horizon"]
    return bounds.StructureDims(
        horizon=horizon,
        has_numeric_horizon=isinstance(horizon, int),
        joint_state_dim=_joint_state_dim(structure, model, validation),
        state_space_dim_total=int(structure["total_state_space_dim"]),
        max_variable_dim=int(structure["max_variable_dim"]),
        max_factor_arity=max(structure["factor_arities"], default=0),
        edge_count=int(structure["edge_count"]),
        variable_count=int(structure["variable_count"]),
        agents=_agent_count(spec),
        regimes=_regime_count(model, validation),
    )


def _joint_state_dim(
    structure: dict[str, Any],
    model: GNNInternalRepresentation,
    validation: dict[str, Any] | None,
) -> int:
    """Joint continuous state dimension ``d`` for dense-LGSSM bounds.

    The first declared dimension of the F/H/Q/R system-matrix variables
    (square ``[d, d]`` declarations); the largest such ``d`` wins.
    Fallback: the max per-variable dim product.
    """
    candidates: list[int] = []
    entries = validation["variables"] if validation is not None else []
    if entries:
        for entry in entries:
            if str(entry["name"]) in _LGSSM_SYSTEM_MATRIX_NAMES:
                dims = entry.get("dimensions") or []
                if dims:
                    candidates.append(int(dims[0]))
    else:
        for variable in model.variables:
            if variable.name in _LGSSM_SYSTEM_MATRIX_NAMES and variable.dimensions:
                candidates.append(int(variable.dimensions[0]))
    return max(candidates) if candidates else int(structure["max_variable_dim"])


def _regime_count(
    model: GNNInternalRepresentation,
    validation: dict[str, Any] | None,
) -> int:
    """Regime multiplier: the declared count of ``B_regime`` tensors.

    The first dimension of each ``B_regime``-style declaration is the
    regime count; the largest wins. 1 when no such declaration exists —
    time-indexed nonstationary specs carry no per-regime tensor multiplier.
    """
    counts: list[int] = []
    entries = validation["variables"] if validation is not None else []
    if entries:
        for entry in entries:
            if _REGIME_VARIABLE.match(str(entry["name"])):
                dims = entry.get("dimensions") or []
                if dims:
                    counts.append(int(dims[0]))
    else:
        for variable in model.variables:
            if _REGIME_VARIABLE.match(variable.name) and variable.dimensions:
                counts.append(int(variable.dimensions[0]))
    return max(counts) if counts else 1


def _agent_count(spec: dict[str, Any]) -> int:
    """Declared agent count, mirroring multi-agent kind detection.

    Priority: an explicit ``nr_agents`` then ``num_agents`` declaration
    (``initialparameterization`` first, then ``model_parameters`` — the
    parser mirrors matrix/parameter keys into both). When neither is
    declared, the count derives from per-agent matrix keys (``A_agent2``
    -> agent 2 declared -> count 2), the same evidence
    ``detect_model_kinds`` uses to classify ``multi_agent``. 1 when no
    agent evidence exists — single-agent models stay 1.
    """
    initial = spec.get("initialparameterization")
    if not isinstance(initial, dict):
        initial = {}
    model_parameters = spec.get("model_parameters")
    if not isinstance(model_parameters, dict):
        model_parameters = {}
    for source in (initial, model_parameters):
        for key in ("nr_agents", "num_agents"):
            try:
                value = int(source[key])
            except (KeyError, TypeError, ValueError):
                continue
            if value > 0:
                return value
    agents: list[int] = []
    for key in initial:
        agent_match = _AGENT_MATRIX_KEY.match(str(key))
        if agent_match is not None:
            agents.append(int(agent_match.group(1)))
    return max(agents) if agents else 1


def _per_backend_rows(
    kinds: frozenset[str] | set[str], dims: bounds.StructureDims
) -> list[dict[str, Any]]:
    """One bound row per backend, in fixed ``BACKEND_ORDER``.

    For frameworks carrying several family variants (``jax``), the first
    applicable variant wins; when none applies, the row reports the
    primary variant with ``applicable: False`` and no drivers.
    """
    kind_set = frozenset(kinds)
    rows: list[dict[str, Any]] = []
    for framework in bounds.BACKEND_ORDER:
        variants = [
            entry for entry in bounds.BACKEND_BOUNDS if entry.framework == framework
        ]
        matched = [
            entry for entry in variants if entry.applicable(kind_set, dims)
        ]
        chosen = matched[0] if matched else variants[0]
        rows.append(
            {
                "framework": framework,
                "applicable": bool(matched),
                "family": chosen.family,
                "asymptotic": chosen.asymptotic(dims),
                "complexity_class": chosen.complexity_class,
                "drivers": _drivers(chosen.family, dims) if matched else {},
                "notes": chosen.notes,
            }
        )
    return rows


def _drivers(family: str, dims: bounds.StructureDims) -> dict[str, Any]:
    """Numeric drivers for one bound family (all values computed, none invented)."""
    if family == "exact-factorized":
        return {
            "horizon": dims.horizon,
            "state_space_dim_total": dims.state_space_dim_total,
            "max_variable_dim": dims.max_variable_dim,
            "max_factor_arity": dims.max_factor_arity,
            "agents": dims.agents,
            "regimes": dims.regimes,
        }
    if family == "exact-dense-LGSSM":
        return {
            "horizon": dims.horizon,
            "joint_state_dim": dims.joint_state_dim,
            "agents": dims.agents,
        }
    if family == "sampling":
        return {
            "horizon": dims.horizon,
            "state_space_dim_total": dims.state_space_dim_total,
            "agents": dims.agents,
        }
    if family == "categorical-composition":
        return {
            "state_space_dim_total": dims.state_space_dim_total,
            "edge_count": dims.edge_count,
        }
    if family == "structure-learning":
        return {
            "variable_count": dims.variable_count,
            "max_variable_dim": dims.max_variable_dim,
        }
    return {}
