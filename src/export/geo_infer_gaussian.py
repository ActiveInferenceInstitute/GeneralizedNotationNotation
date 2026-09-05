"""Export explicit GNN linear Gaussian models to the data-only v2 contract."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from .geo_infer import MAX_MATRIX_ENTRIES, MAX_SOURCE_BYTES

CONTRACT_VERSION = "gnn-geo-infer/2"


def _keys(value: Any, names: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError(f"{context} requires exactly {sorted(names)}")


def _shape_matches(value: Any, shape: tuple[int, ...]) -> bool:
    if not shape:
        return type(value) in (int, float)
    return (
        isinstance(value, list)
        and len(value) == shape[0]
        and all(_shape_matches(x, shape[1:]) for x in value)
    )


def validate_gaussian_artifact(data: dict[str, Any]) -> None:
    """Reject unsupported axes, generators, invalid covariances and missing units."""
    _keys(
        data,
        {
            "schema_version",
            "model_type",
            "model_name",
            "dimensions",
            "matrices",
            "initial_belief",
            "units",
            "time",
            "provenance",
        },
        "artifact",
    )
    if (
        data["schema_version"] != CONTRACT_VERSION
        or data["model_type"] != "linear_gaussian"
    ):
        raise ValueError("Unsupported Gaussian contract version or model type")
    if not isinstance(data["model_name"], str) or not data["model_name"].strip():
        raise ValueError("model_name must be nonempty")
    dims = data["dimensions"]
    _keys(dims, {"states", "observations", "controls"}, "dimensions")
    if any(type(v) is not int or v < 1 for v in dims.values()):
        raise ValueError("Dimensions must be positive integers")
    n, m, k = (dims[x] for x in ("states", "observations", "controls"))
    if 3 * n * n + n * k + m * n + m * m + n > MAX_MATRIX_ENTRIES:
        raise ValueError("Artifact exceeds dense matrix entry budget")
    shapes = {"F": (n, n), "G": (n, k), "H": (m, n), "Q": (n, n), "R": (m, m)}
    _keys(data["matrices"], set(shapes), "matrices")
    _keys(data["initial_belief"], {"mean", "covariance"}, "initial_belief")
    values = dict(
        data["matrices"],
        mean=data["initial_belief"]["mean"],
        covariance=data["initial_belief"]["covariance"],
    )
    arrays = {}
    for name, shape in dict(shapes, mean=(n,), covariance=(n, n)).items():
        if not _shape_matches(values[name], shape):
            raise ValueError(f"{name} requires JSON numbers with shape {shape}")
        array = np.asarray(values[name], dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        arrays[name] = array
    for name in ("Q", "R", "covariance"):
        array = arrays[name]
        if not np.allclose(array, array.T, rtol=0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(array * 0.5 + array.T * 0.5)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"{name} covariance spectrum could not be validated"
            ) from exc
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError(f"{name} covariance spectrum must be finite")
        if np.min(eigenvalues) < 0 or (name != "Q" and np.min(eigenvalues) <= 0):
            raise ValueError(
                f"{name} must be positive {'semidefinite' if name == 'Q' else 'definite'}"
            )
    _keys(data["units"], set(dims), "units")
    for name, size in dims.items():
        units = data["units"][name]
        if (
            not isinstance(units, list)
            or len(units) != size
            or any(not isinstance(x, str) or not x.strip() for x in units)
        ):
            raise ValueError(f"units.{name} must label each coordinate")
    _keys(data["time"], {"domain", "step_seconds"}, "time")
    seconds = data["time"]["step_seconds"]
    if data["time"]["domain"] != "discrete":
        raise ValueError(
            "Only explicit discrete-time transitions are supported; generators require discretization"
        )
    if type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    _keys(data["provenance"], {"producer", "source_sha256"}, "provenance")
    provenance = data["provenance"]
    if (
        not isinstance(provenance["producer"], str)
        or not provenance["producer"].strip()
    ):
        raise ValueError("producer must be nonempty")
    if not isinstance(provenance["source_sha256"], str) or not re.fullmatch(
        "[0-9a-f]{64}", provenance["source_sha256"]
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def build_geo_infer_gaussian_artifact(
    content: str,
    *,
    step_seconds: float,
    units: dict[str, list[str]],
    time_domain: str = "discrete",
) -> dict[str, Any]:
    """Extract F/G/H/Q/R and prior_mean/prior_cov without inserting defaults.

    The caller must declare one unit per state, observed and control coordinate.
    GNN Time must explicitly contain Discrete and must not contain Continuous.
    Continuous state does not imply continuous time: F and Q are per interval.
    """
    from gnn.pomdp_extractor import GNNExtractionError, extract_pomdp_from_content

    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("GNN source must be text within four MiB")
    time_sections = re.findall(
        r"^##[ \t]+Time[ \t]*$\n(.*?)(?=^##[ \t]|\Z)", content, flags=re.M | re.S
    )
    declarations = (
        set()
        if len(time_sections) != 1
        else {
            line.split("#", 1)[0].strip().lower()
            for line in time_sections[0].splitlines()
        }
    )
    if (
        time_domain != "discrete"
        or "discrete" not in declarations
        or "continuous" in declarations
    ):
        raise ValueError(
            "GNN Time must explicitly declare Discrete; generators are unsupported"
        )
    try:
        model = extract_pomdp_from_content(
            content, strict_validation=True, on_error="raise", insert_default_c=False
        )
    except GNNExtractionError as exc:
        raise ValueError(f"Invalid Gaussian GNN source: {exc}") from exc
    if model is None or model.model_kind != "continuous":
        raise ValueError("Gaussian export requires a linear Gaussian GNN model")
    parameters = json.loads(
        json.dumps(model.initial_parameterization or {}, allow_nan=False)
    )
    required = {"F", "G", "H", "Q", "R", "prior_mean", "prior_cov"}
    if not required.issubset(parameters):
        raise ValueError(
            f"Explicit Gaussian parameters required: {sorted(required - set(parameters))}"
        )
    control = parameters["G"]
    if (
        not isinstance(control, list)
        or not control
        or not isinstance(control[0], list)
        or not control[0]
    ):
        raise ValueError("G must be a nonempty state-by-control matrix")
    data = dict(
        schema_version=CONTRACT_VERSION,
        model_type="linear_gaussian",
        model_name=model.model_name or "GNN Gaussian model",
        dimensions=dict(
            states=model.num_states,
            observations=model.num_observations,
            controls=len(control[0]),
        ),
        matrices={name: parameters[name] for name in ("F", "G", "H", "Q", "R")},
        initial_belief=dict(
            mean=parameters["prior_mean"], covariance=parameters["prior_cov"]
        ),
        units=units,
        time=dict(domain=time_domain, step_seconds=step_seconds),
        provenance=dict(
            producer="GNN strict Gaussian extractor",
            source_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ),
    )
    validate_gaussian_artifact(data)
    _validate_source_declarations(content, data, parameters)
    return data


def _validate_source_declarations(
    content: str, data: dict[str, Any], parameters: dict[str, Any]
) -> None:
    """Check explicit GNN coordinate and parameter declarations against values."""
    from gnn.pomdp_extractor import POMDPExtractor

    sections = re.findall(
        r"^##[ \t]+StateSpaceBlock[ \t]*$\n(.*?)(?=^##[ \t]|\Z)",
        content,
        flags=re.M | re.S,
    )
    if len(sections) != 1:
        raise ValueError("Gaussian source requires exactly one StateSpaceBlock")
    declarations = {}
    variable_pattern = POMDPExtractor().VARIABLE_PATTERN
    for line in sections[0].splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = variable_pattern.fullmatch(line)
        if not match:
            raise ValueError(f"Unsupported Gaussian variable declaration: {line}")
        name = match.group(1)
        if name in declarations:
            raise ValueError(f"Duplicate Gaussian variable: {name}")
        tokens = [token.strip() for token in match.group(2).split(",")]
        declared_types = [
            token.split("=", 1)[1].strip()
            for token in tokens
            if token.startswith("type=")
        ]
        expected_type = "int" if name == "t" else "float"
        if any(kind != expected_type for kind in declared_types):
            raise ValueError(f"{name} declaration requires type={expected_type}")
        try:
            dimensions = tuple(int(token) for token in tokens if "=" not in token)
        except ValueError as exc:
            raise ValueError(f"{name} requires literal positive dimensions") from exc
        if not dimensions or any(size < 1 for size in dimensions):
            raise ValueError(f"{name} requires literal positive dimensions")
        declarations[name] = dimensions
    for name, dimension in (("x", "states"), ("y", "observations"), ("u", "controls")):
        size = data["dimensions"][dimension]
        if declarations.get(name) not in ((size,), (size, 1)):
            raise ValueError(f"{name} declaration must match {dimension}={size}")
    for name, dimensions in declarations.items():
        if name in {"x", "y", "u", "t"}:
            continue
        if name not in parameters or not _shape_matches(parameters[name], dimensions):
            raise ValueError(
                f"Unsupported or contradictory Gaussian declaration: {name}"
            )
