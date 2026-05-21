#!/usr/bin/env python3
"""Canonical POMDP render contract shared by framework renderers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple, cast

import numpy as np

CANONICAL_B_ORDER = "next_state_previous_state_action"


def nested_shape(value: Any) -> List[int]:
    """Return the shape of a regular nested list-like value."""
    shape: List[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def normalise_vector(value: Any, *, name: str) -> List[float]:
    """Return a finite probability vector."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return cast(List[float], (vector / total).astype(float).tolist())


def normalise_matrix_columns(value: Any, *, name: str) -> List[List[float]]:
    """Return a 2-D matrix with columns summing to one."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {matrix.shape}")
    if min(matrix.shape) <= 0:
        raise ValueError(f"{name} must not be empty")
    output = matrix.copy()
    for column in range(output.shape[1]):
        total = float(output[:, column].sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"{name} column {column} has invalid probability mass")
        output[:, column] /= total
    return cast(List[List[float]], output.astype(float).tolist())


def _declared_b_order(model_parameters: Dict[str, Any]) -> str:
    order = str(
        model_parameters.get("b_tensor_order")
        or model_parameters.get("B_tensor_order")
        or model_parameters.get("transition_tensor_order")
        or ""
    ).lower()
    return order.replace("-", "_").replace(" ", "_")


def canonicalise_b_matrix(
    value: Any,
    *,
    num_states: int,
    num_actions: int,
    model_parameters: Dict[str, Any],
) -> Tuple[List[List[List[float]]], Dict[str, Any]]:
    """Return B as ``(next_state, previous_state, action)``."""
    raw = np.asarray(value, dtype=np.float64)
    declared_order = _declared_b_order(model_parameters)
    source_order = "inferred"

    if raw.ndim == 2:
        if raw.shape != (num_states, num_states):
            raise ValueError(
                f"B must be ({num_states}, {num_states}) for passive dynamics, got {raw.shape}"
            )
        tensor = raw[:, :, np.newaxis]
        source_order = "next_state_previous_state"
    elif raw.ndim == 3:
        if declared_order in {
            CANONICAL_B_ORDER,
            "next_previous_action",
            "next_prev_action",
            "states_next_states_previous_actions",
        }:
            tensor = raw
            source_order = CANONICAL_B_ORDER
        elif declared_order in {
            "action_next_state_previous_state",
            "action_next_previous",
            "actions_next_previous",
        }:
            tensor = raw.transpose(1, 2, 0)
            source_order = "action_next_state_previous_state"
        elif declared_order in {
            "action_previous_state_next_state",
            "action_previous_next",
            "actions_previous_next",
        }:
            tensor = raw.transpose(2, 1, 0)
            source_order = "action_previous_state_next_state"
        elif raw.shape == (num_actions, num_states, num_states):
            tensor = raw.transpose(2, 1, 0)
            source_order = "action_previous_state_next_state"
        elif raw.shape == (num_states, num_states, num_actions):
            tensor = raw
            source_order = CANONICAL_B_ORDER
        elif raw.shape == (1, num_states, num_states):
            tensor = raw.transpose(2, 1, 0)
            source_order = "single_action_previous_state_next_state"
        else:
            raise ValueError(
                "B must be 2-D passive, canonical "
                f"({num_states}, {num_states}, {num_actions}), or action-first "
                f"({num_actions}, {num_states}, {num_states}); got {raw.shape}"
            )
    else:
        raise ValueError(f"B must be 2-D or 3-D, got shape {raw.shape}")

    if tensor.shape[0] != num_states or tensor.shape[1] != num_states:
        raise ValueError(
            f"Canonical B shape {tensor.shape} does not match {num_states} states"
        )
    if tensor.shape[2] not in {1, num_actions}:
        raise ValueError(
            f"Canonical B action dimension {tensor.shape[2]} does not match {num_actions}"
        )

    output = tensor.astype(np.float64).copy()
    for action in range(output.shape[2]):
        for previous_state in range(output.shape[1]):
            total = float(output[:, previous_state, action].sum())
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "B[:, previous_state, action] must have positive finite mass "
                    f"for previous_state={previous_state}, action={action}"
                )
            output[:, previous_state, action] /= total

    return (
        cast(List[List[List[float]]], output.astype(float).tolist()),
        {
            "source_order": source_order,
            "canonical_order": CANONICAL_B_ORDER,
            "shape": list(output.shape),
        },
    )


def _first_int(*values: Any, required: bool = True) -> int:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    if required:
        raise ValueError("required positive integer dimension is missing")
    return 0


def _infer_num_actions_from_b(value: Any, *, num_states: int) -> int:
    """Infer the control dimension from a passive or controlled B matrix."""
    shape = nested_shape(value)
    if len(shape) == 2:
        return 1 if shape == [num_states, num_states] else 0
    if len(shape) != 3:
        return 0
    if shape[0] == num_states and shape[1] == num_states:
        return shape[2]
    if shape[1] == num_states and shape[2] == num_states:
        return shape[0]
    return 0


def _require_keys(mapping: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required initialparameterization keys: {missing}")


def build_canonical_pomdp_spec(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copied GNN spec with canonical POMDP matrices and provenance."""
    spec = deepcopy(gnn_spec)
    initial = spec.get("initialparameterization") or spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        raise ValueError("GNN spec requires initialparameterization")
    _require_keys(initial, ("A", "B", "C", "D"))

    model_parameters = dict(spec.get("model_parameters") or {})
    num_states = _first_int(
        model_parameters.get("num_hidden_states"),
        model_parameters.get("num_states"),
        spec.get("num_states"),
        len(initial.get("D") or []),
    )
    num_observations = _first_int(
        model_parameters.get("num_obs"),
        model_parameters.get("num_observations"),
        spec.get("num_observations"),
        len(initial.get("A") or []),
    )
    num_actions = _first_int(
        model_parameters.get("num_actions"),
        model_parameters.get("num_controls"),
        spec.get("num_actions"),
        len(initial.get("E") or []),
        _infer_num_actions_from_b(initial["B"], num_states=num_states),
    )

    canonical_b, b_provenance = canonicalise_b_matrix(
        initial["B"],
        num_states=num_states,
        num_actions=num_actions,
        model_parameters=model_parameters,
    )
    action_count = len(canonical_b[0][0]) if canonical_b else num_actions

    canonical_initial: Dict[str, Any] = {
        "A": normalise_matrix_columns(initial["A"], name="A"),
        "B": canonical_b,
        "C": np.asarray(initial["C"], dtype=np.float64)
        .reshape(-1)
        .astype(float)
        .tolist(),
        "D": normalise_vector(initial["D"], name="D"),
    }
    if "E" in initial:
        canonical_initial["E"] = normalise_vector(initial["E"], name="E")

    model_parameters.update(
        {
            "num_hidden_states": num_states,
            "num_obs": num_observations,
            "num_actions": action_count,
            "b_tensor_order": CANONICAL_B_ORDER,
        }
    )
    spec["model_parameters"] = model_parameters
    spec["initialparameterization"] = canonical_initial
    spec["initial_parameterization"] = canonical_initial

    provenance = dict(spec.get("matrix_provenance") or {})
    existing_b = dict(provenance.get("B") or {})
    provenance["B"] = {
        **existing_b,
        **b_provenance,
        "derived": existing_b.get("derived", False),
    }
    for key in ("A", "C", "D", "E"):
        if key in canonical_initial and key not in provenance:
            provenance[key] = {
                "source": "initialparameterization",
                "shape": nested_shape(canonical_initial[key]),
                "derived": False,
            }
    spec["matrix_provenance"] = provenance

    structured = dict(spec.get("structured_pomdp") or {})
    structured["matrix_provenance"] = provenance
    structured["canonical_b_order"] = CANONICAL_B_ORDER
    spec["structured_pomdp"] = structured
    spec["canonical_pomdp_schema"] = "canonical_pomdp_v1"
    return spec
