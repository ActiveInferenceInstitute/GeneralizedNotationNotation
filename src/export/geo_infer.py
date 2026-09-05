"""Export strict GNN categorical models to the versioned GEO-INFER contract.

This data-only boundary does not import GEO-INFER or execute generated source.
Use ``python -m export.geo_infer MODEL.md OUTPUT.json --step-seconds 1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = "gnn-geo-infer/1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_MATRIX_ENTRIES = 1_000_000


def build_geo_infer_artifact(
    content: str,
    *,
    step_seconds: float,
    state_ids: list[str] | None = None,
    space_kind: str = "categorical",
) -> dict[str, Any]:
    """Extract explicit A–E matrices and attach caller-declared physical time.

    Args:
        content: GNN Markdown source (at most four MiB UTF-8).
        step_seconds: Positive seconds represented by one B transition.
        state_ids: Unique IDs in the existing matrix state order; never reorders B.
        space_kind: ``categorical`` or ``h3`` (H3 validity checked by GEO consumer).
    Returns:
        A JSON-safe v1 artifact with exact extracted values and source digest.
    Raises:
        ValueError: Unsupported model, derived/missing matrices, or invalid semantics.
    Example:
        artifact = build_geo_infer_artifact(source_text, step_seconds=60)
    """
    from gnn.pomdp_extractor import extract_pomdp_from_content

    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("GNN source must be text within four MiB")
    if (
        isinstance(step_seconds, bool)
        or not isinstance(step_seconds, (float, int))
        or not math.isfinite(step_seconds)
        or step_seconds <= 0
    ):
        raise ValueError("step_seconds must be finite and positive")
    model = extract_pomdp_from_content(
        content, strict_validation=True, on_error="raise", insert_default_c=False
    )
    if (
        model is None
        or model.model_kind != "discrete"
        or (
            model.num_state_factors,
            model.num_observation_modalities,
            model.num_control_factors,
        )
        != (1, 1, 1)
    ):
        raise ValueError(
            "GEO-INFER v1 requires one categorical state factor, modality and control factor"
        )
    s, o, u = model.num_states, model.num_observations, model.num_actions
    if o * s + s * s * u + o + s + u > MAX_MATRIX_ENTRIES:
        raise ValueError("Model exceeds dense matrix entry budget")
    values = dict(
        A=model.A_matrix,
        B=model.B_matrix,
        C=model.C_vector,
        D=model.D_vector,
        E=model.E_vector,
    )
    provenance = model.matrix_provenance or {}
    shapes = dict(A=(o, s), B=(s, s, u), C=(o,), D=(s,), E=(u,))
    matrices = {}
    for name, shape in shapes.items():
        origin = provenance.get(name, {})
        if (
            values[name] is None
            or origin.get("derived", True)
            or origin.get("source") != "InitialParameterization"
        ):
            raise ValueError(
                f"{name} must be explicitly supplied in InitialParameterization"
            )
        if name == "B" and (
            origin.get("contradiction")
            or origin.get("detected_order")
            != ["next_state", "previous_state", "action"]
        ):
            raise ValueError("B has contradictory axis conventions")
        matrix = np.asarray(values[name], dtype=float)
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} requires finite values with shape {shape}")
        if name != "C" and (
            np.any(matrix < 0)
            or not np.allclose(matrix.sum(axis=0), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                f"{name} must be nonnegative and stochastic along axis zero"
            )
        matrices[name] = matrix.tolist()
    ids = [str(i) for i in range(s)] if state_ids is None else list(state_ids)
    if (
        len(ids) != s
        or any(not isinstance(x, str) or not x for x in ids)
        or len(set(ids)) != s
    ):
        raise ValueError("state_ids must uniquely label every state in matrix order")
    if space_kind not in {"categorical", "h3"} or (
        space_kind == "h3" and state_ids is None
    ):
        raise ValueError("H3 space requires explicit H3 state IDs")
    if space_kind == "h3":
        try:
            import h3
        except ImportError as exc:
            raise ValueError(
                "H3 export requires h3>=4.5,<5 in the GNN environment"
            ) from exc
        if (
            any(
                not h3.is_valid_cell(x) or h3.int_to_str(h3.str_to_int(x)) != x
                for x in ids
            )
            or len({h3.get_resolution(x) for x in ids}) != 1
        ):
            raise ValueError("H3 state IDs must be canonical cells at one resolution")
    return dict(
        schema_version=CONTRACT_VERSION,
        model_type="categorical",
        model_name=model.model_name or "GNN model",
        dimensions=dict(states=s, observations=o, actions=u),
        matrices=matrices,
        space=dict(kind=space_kind, state_ids=ids),
        time=dict(step_seconds=step_seconds),
        provenance=dict(
            producer="GNN strict POMDP extractor",
            source_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ),
    )


def export_to_geo_infer(model_data: dict[str, Any], output_file: Path) -> bool:
    """Write an opt-in artifact from raw_content and explicit geo_infer options.

    Args:
        model_data: Mapping with ``raw_content`` and ``geo_infer`` keyword options.
        output_file: Destination JSON file. Its parent must already exist.
    Returns:
        True after a validated artifact has been written.
    Raises:
        ValueError: Source text or explicit timestep is absent, or model is unsupported.
    Example:
        export_to_geo_infer({'raw_content': text, 'geo_infer': {'step_seconds': 1}}, path)
    """
    options = model_data.get("geo_infer")
    if (
        not isinstance(options, dict)
        or "step_seconds" not in options
        or "raw_content" not in model_data
    ):
        raise ValueError(
            "geo_infer export requires raw_content and explicit geo_infer.step_seconds"
        )
    options = dict(options)
    model_type = options.pop("model_type", "categorical")
    if model_type == "linear_gaussian":
        from .geo_infer_gaussian import build_geo_infer_gaussian_artifact

        artifact = build_geo_infer_gaussian_artifact(
            model_data["raw_content"], **options
        )
    elif model_type == "categorical":
        artifact = build_geo_infer_artifact(model_data["raw_content"], **options)
    else:
        raise ValueError(f"Unsupported geo_infer model_type: {model_type}")
    output_file.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    """Export a GNN source file; return zero after writing its validated artifact.

    Arguments are described by ``python -m export.geo_infer --help``.
    Raises ValueError for invalid or oversized source models.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-seconds", type=float, required=True)
    parser.add_argument(
        "--state-ids", type=Path, help="JSON array in matrix state order"
    )
    parser.add_argument(
        "--space-kind", choices=["categorical", "h3"], default="categorical"
    )
    parser.add_argument(
        "--model-type",
        choices=["categorical", "linear_gaussian"],
        default="categorical",
    )
    parser.add_argument(
        "--units",
        type=Path,
        help="Gaussian JSON object with states, observations and controls unit arrays",
    )
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("GNN source exceeds four MiB")
    options: dict[str, Any] = dict(
        step_seconds=args.step_seconds, space_kind=args.space_kind
    )
    if args.model_type == "linear_gaussian":
        if args.units is None or args.state_ids or args.space_kind != "categorical":
            raise ValueError(
                "Gaussian export requires --units and does not accept categorical space options"
            )
        with args.units.open("rb") as stream:
            raw_units = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw_units) > MAX_SOURCE_BYTES:
            raise ValueError("Units exceed four MiB")
        options = dict(
            model_type="linear_gaussian",
            step_seconds=args.step_seconds,
            units=json.loads(raw_units),
        )
    elif args.units is not None:
        raise ValueError("--units requires --model-type linear_gaussian")
    if args.state_ids:
        with args.state_ids.open("rb") as stream:
            labels = stream.read(MAX_SOURCE_BYTES + 1)
        if len(labels) > MAX_SOURCE_BYTES:
            raise ValueError("State IDs exceed four MiB")
        options["state_ids"] = json.loads(labels)
        if not isinstance(options["state_ids"], list):
            raise ValueError("State IDs file must be a JSON array")
    export_to_geo_infer(
        dict(raw_content=raw.decode("utf-8"), geo_infer=options), args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
