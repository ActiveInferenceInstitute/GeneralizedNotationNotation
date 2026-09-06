"""Export explicitly structured factored categorical models for GEO-INFER.

Input is JSON with declared dependency axes and policy enumeration. This API
is not a Markdown factor-axis extractor and does not derive missing matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "gnn-geo-infer/factored/1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 1_000_000
MAX_JOINT_STATES = 256
MAX_POLICY_WORK = 20_000_000


def _keys(value: Any, expected: str, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{label} requires exactly {expected}")


def _integer(value: Any, upper: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise ValueError(f"{label} must be an integer in [0, {upper})")
    return int(value)


def _labels(values: Any, label: str) -> int:
    if (
        not isinstance(values, list)
        or not values
        or len(values) > 256
        or any(not isinstance(v, str) or not v for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{label} must contain 1..256 unique nonempty labels")
    return len(values)


def _shape(value: Any, dimensions: list[int], label: str) -> None:
    if not dimensions:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must contain finite numeric values")
        return
    if not isinstance(value, list) or len(value) != dimensions[0]:
        raise ValueError(f"{label} requires shape {dimensions}")
    for child in value:
        _shape(child, dimensions[1:], label)


def validate_factored_artifact(
    data: dict[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    """Validate dependency axes and bounds before allocating numerical tensors."""
    _keys(
        data,
        "schema_version model_type model_name state_factors control_factors modalities transitions initial_joint policies policy_prior time provenance",
        "Artifact",
    )
    if (
        data["schema_version"] != CONTRACT_VERSION
        or data["model_type"] != "categorical_factored"
    ):
        raise ValueError("Unsupported factored contract version or model type")
    if not isinstance(data["model_name"], str) or not data["model_name"]:
        raise ValueError("model_name must be nonempty")
    _keys(data["time"], "step_seconds", "time")
    seconds = data["time"]["step_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("step_seconds must be finite and positive")
    _keys(data["provenance"], "producer source_kind source_sha256", "provenance")
    origin = data["provenance"]
    if (
        origin["source_kind"] != "explicit_factored_json"
        or not isinstance(origin["producer"], str)
        or not origin["producer"]
        or not isinstance(origin["source_sha256"], str)
        or len(origin["source_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in origin["source_sha256"])
    ):
        raise ValueError("Invalid structured-source provenance")
    dimensions = []
    for section, labels in (
        ("state_factors", "states"),
        ("control_factors", "actions"),
    ):
        factors = data[section]
        if not isinstance(factors, list) or not 1 <= len(factors) <= 8:
            raise ValueError(f"{section} must contain 1..8 factors")
        sizes, ids = [], []
        for factor in factors:
            _keys(factor, f"id {labels}", section)
            ids.append(factor["id"])
            sizes.append(_labels(factor[labels], labels))
        _labels(ids, section)
        dimensions.append(sizes)
    states, controls = dimensions
    joint_states = math.prod(states)
    if joint_states > MAX_JOINT_STATES:
        raise ValueError("Joint state budget exceeded")
    entries = joint_states * joint_states + joint_states
    arrays = [("initial_joint", data["initial_joint"], [joint_states], True)]
    modalities = data["modalities"]
    transitions = data["transitions"]
    if not isinstance(modalities, list) or not 1 <= len(modalities) <= 8:
        raise ValueError("modalities must contain 1..8 observation modalities")
    if not isinstance(transitions, list) or len(transitions) != len(states):
        raise ValueError("One transition tensor is required per state factor")
    outcome_sizes, modality_ids = [], []
    for index, modality in enumerate(modalities):
        _keys(modality, "id outcomes dependencies likelihood preferences", "modality")
        modality_ids.append(modality["id"])
        count = _labels(modality["outcomes"], "outcomes")
        outcome_sizes.append(count)
        arrays.append((f"C[{index}]", modality["preferences"], [count], False))
    _labels(modality_ids, "modality IDs")
    for label, items in (("A", modalities), ("B", transitions)):
        for index, item in enumerate(items):
            if label == "B":
                _keys(item, "dependencies control_factor probabilities", "transition")
            deps = item["dependencies"]
            if not isinstance(deps, list) or len(deps) > len(states):
                raise ValueError(f"{label} dependencies must list state factor indices")
            for dep in deps:
                _integer(dep, len(states), "dependency")
            if len(set(deps)) != len(deps):
                raise ValueError("Dependency axes must be unique")
            shape = [outcome_sizes[index] if label == "A" else states[index]]
            shape += [states[dep] for dep in deps]
            if label == "B":
                control = _integer(
                    item["control_factor"], len(controls), "control_factor"
                )
                shape.append(controls[control])
            arrays.append(
                (
                    f"{label}[{index}]",
                    item["likelihood" if label == "A" else "probabilities"],
                    shape,
                    True,
                )
            )
    outcomes = math.prod(outcome_sizes)
    entries += outcomes * joint_states
    policies = data["policies"]
    if not isinstance(policies, list) or not 1 <= len(policies) <= 256:
        raise ValueError("policies must explicitly enumerate 1..256 policies")
    horizon = len(policies[0]) if isinstance(policies[0], list) else 0
    if not 1 <= horizon <= 8:
        raise ValueError("Policy horizon must be 1..8")
    for policy in policies:
        if not isinstance(policy, list) or len(policy) != horizon:
            raise ValueError("All policies must have the same horizon")
        for action in policy:
            if not isinstance(action, list) or len(action) != len(controls):
                raise ValueError("Every policy action must specify each control factor")
            for value, count in zip(action, controls):
                _integer(value, count, "action")
    if len({json.dumps(p) for p in policies}) != len(policies):
        raise ValueError("Policies must be unique")
    arrays.append(("E", data["policy_prior"], [len(policies)], True))
    nodes, level_nodes = 0, 1
    for _ in range(horizon):
        nodes += level_nodes
        if (
            len(policies) * nodes * joint_states * joint_states * outcomes
            > MAX_POLICY_WORK
        ):
            raise ValueError("Exact policy observation-tree work budget exceeded")
        level_nodes *= outcomes
    entries += sum(math.prod(shape) for _, _, shape, _ in arrays)
    if entries > MAX_ENTRIES:
        raise ValueError("Factored matrix entry budget exceeded")
    for label, value, shape, _ in arrays:
        _shape(value, shape, label)
    # Numerical conversion follows every dimension, budget, and nesting check.
    for label, value, _, probability in arrays:
        array = np.asarray(value, dtype=float)
        if probability and (
            np.any(array < 0)
            or not np.allclose(array.sum(axis=0), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                f"{label} must be nonnegative and stochastic along axis zero"
            )
    return states, controls, outcome_sizes


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def build_geo_infer_factored_artifact(
    content: str, *, step_seconds: float
) -> dict[str, Any]:
    """Validate explicit JSON input and retain its exact UTF-8 source digest."""
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Factored source must be UTF-8 text within four MiB")
    source = json.loads(content, object_pairs_hook=_unique_object)
    _keys(
        source,
        "model_name state_factors control_factors modalities transitions initial_joint policies policy_prior",
        "Structured factored source",
    )
    artifact = {
        **source,
        "schema_version": CONTRACT_VERSION,
        "model_type": "categorical_factored",
        "time": {"step_seconds": step_seconds},
        "provenance": {
            "producer": "GNN explicit factored JSON exporter",
            "source_kind": "explicit_factored_json",
            "source_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        },
    }
    validate_factored_artifact(artifact)
    return artifact


def main() -> int:
    """Export one explicit factored JSON model without importing GEO-INFER."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-seconds", type=float, required=True)
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("Factored source exceeds four MiB")
    artifact = build_geo_infer_factored_artifact(
        raw.decode("utf-8"), step_seconds=args.step_seconds
    )
    args.output.write_text(
        json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
