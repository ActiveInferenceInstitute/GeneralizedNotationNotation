"""Strict source-declared finite controls; no inference dependency imports."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from .admission import MAX_ESTIMATED_BYTES, MAX_POLICIES

FINITE_CONTROL_CONTRACT = "gnn-cpomdp-finite-actions-v1"


def finite_controls(raw: object, n: int) -> dict[str, Any]:
    """Validate ordered actions and an explicit state-by-control linear map.

    Duplicate vectors retain distinct action IDs and authored tie order. Units
    are declarations, not a dimensional-analysis or extraction proof.
    """
    keys = {
        "contract",
        "version",
        "action_ids",
        "control_units",
        "state_units",
        "actions",
        "control_matrix",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"cpomdp_control requires exactly {sorted(keys)}")
    if raw["contract"] != FINITE_CONTROL_CONTRACT:
        raise ValueError("unsupported cpomdp_control contract")

    def label(value: object) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError(
                "cpomdp_control labels must be nonempty strings of at most 128 characters"
            )
        return value

    version = label(raw["version"])
    actions, matrix = raw["actions"], raw["control_matrix"]
    if not isinstance(actions, (list, tuple)) or not 2 <= len(actions) <= MAX_POLICIES:
        raise ValueError("cpomdp_control requires 2..4096 ordered actions")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != n:
        raise ValueError("cpomdp_control control_matrix must have n state rows")
    if not isinstance(matrix[0], (list, tuple)) or not matrix[0]:
        raise ValueError("cpomdp_control control_matrix needs a nonempty control axis")
    p = len(matrix[0])
    if 64 * (len(actions) * p + n * p) > MAX_ESTIMATED_BYTES:
        raise ValueError(
            "cpomdp_control declared allocation exceeds max_estimated_bytes"
        )

    def labels(value: object, count: int) -> list[str]:
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise ValueError(
                "cpomdp_control label dimensions must match their declared axes"
            )
        return [label(v) for v in value]

    ids = labels(raw["action_ids"], len(actions))
    if len(set(ids)) != len(ids):
        raise ValueError("cpomdp_control action_ids must be unique")

    def rows(value: Any) -> list[list[float]]:
        result = []
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != p:
                raise ValueError("cpomdp_control rows must match the control dimension")
            converted = []
            for v in row:
                if isinstance(v, bool) or not isinstance(v, Real):
                    raise ValueError(
                        "cpomdp_control values must be finite nonboolean numbers"
                    )
                try:
                    number = float(v)
                except OverflowError as exc:
                    raise ValueError(
                        "cpomdp_control values must be finite nonboolean numbers"
                    ) from exc
                if not math.isfinite(number):
                    raise ValueError(
                        "cpomdp_control values must be finite nonboolean numbers"
                    )
                converted.append(number)
            result.append(converted)
        return result

    return {
        "contract": FINITE_CONTROL_CONTRACT,
        "version": version,
        "action_ids": ids,
        "control_units": labels(raw["control_units"], p),
        "state_units": labels(raw["state_units"], n),
        "actions": rows(actions),
        "control_matrix": rows(matrix),
    }
