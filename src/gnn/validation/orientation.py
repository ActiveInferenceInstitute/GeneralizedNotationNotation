"""B-tensor orientation diagnostics for Step 6 validation.

Canonical contract (``docs/gnn/gnn_syntax.md``, Initial Parameterization;
``CANONICAL_B_ORDER`` in ``gnn.extract.pomdp_extractor``): transition
tensors are stored ``B[next_state, previous_state, action]`` (= pymdp 1.0.0
``B[s',s,a]``). Each per-action slice is written rows = next states,
columns = previous states and is **column-stochastic**: every column (one
previous state) sums to 1 over next states.

Textbook POMDP literature frequently writes transition matrices the other
way around (rows = previous state ``s_t``, columns = next state
``s_{t+1}``; each row sums to 1). Imported verbatim, such a file is
silently transposed by every canonical reader: ``B[next][prev][action]``
data read as if rows were previous states flips every transition. Step 6
warns about the detected orientation and offers an opt-in canonical
transposition (``--transpose-b``) recorded per tensor in the receipt.

The stochasticity tolerance is the type checker's
``STOCHASTICITY_TOLERANCE`` (the same constant behind the Step 5
``[GNN-E002]`` orientation checks); this module never re-derives it.
Slices that are neither row- nor column-stochastic stay silent here —
the existing stochasticity error paths own that failure.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeAlias

from gnn.type_checker.checking import (
    STOCHASTICITY_TOLERANCE,
    extract_b_matrix_evidence,
    extract_gnn_dimensions,
    numeric_matrix,
)

from .structure import clamp01, display_file_name, extract_content_from_dict

logger = logging.getLogger(__name__)

ModelData: TypeAlias = Mapping[str, Any]
ModelInput: TypeAlias = str | Path | ModelData | None

# Tensor orientations reported per B tensor.
_CANONICAL = "canonical"
_ROW_STOCHASTIC = "row_stochastic"
_AMBIGUOUS = "ambiguous"
_NON_STOCHASTIC = "non_stochastic"

# Per-slice margin classes (row/column sums against 1.0).
_PER_SLICE_COL = "col"
_PER_SLICE_ROW = "row"
_PER_SLICE_BOTH = "both"
_PER_SLICE_NEITHER = "neither"

# Candidate action-axis positions in the stored literal.
_AXIS_INNER = "inner"  # canonical declaration order (next, prev, action)
_AXIS_OUTER = "outer"  # action-outer storage (e.g. (action, prev, next))

Matrix = list[list[float]]
MatrixStack = list[Matrix]


def _nested_shape(values: Any) -> list[int]:
    """Best-effort nested shape of a parsed literal."""
    shape: list[int] = []
    current: Any = values
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def _classify_slice(matrix: Matrix) -> str:
    """Classify one 2-D slice by its row/column margins against 1.0."""
    row_sums = [sum(row) for row in matrix]
    col_sums = [
        sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))
    ]

    def close_to_one(sums: list[float]) -> bool:
        return all(abs(total - 1.0) <= STOCHASTICITY_TOLERANCE for total in sums)

    row_ok = close_to_one(row_sums)
    col_ok = close_to_one(col_sums)
    if row_ok and col_ok:
        return _PER_SLICE_BOTH
    if col_ok:
        return _PER_SLICE_COL
    if row_ok:
        return _PER_SLICE_ROW
    return _PER_SLICE_NEITHER


def _reading_kind(slice_classes: list[str]) -> str:
    """Classify one candidate reading by its per-slice classes.

    Silence-first ladder: canonical (all slices column-stochastic, at least
    one decisively) beats ambiguous (all doubly stochastic — orientation
    indeterminate) beats row (at least one row-stochastic-only slice, none
    column-decisive). Anything else is ``other``.
    """
    if all(c in (_PER_SLICE_COL, _PER_SLICE_BOTH) for c in slice_classes) and any(
        c == _PER_SLICE_COL for c in slice_classes
    ):
        return _CANONICAL
    if all(c == _PER_SLICE_BOTH for c in slice_classes):
        return _AMBIGUOUS
    if all(c in (_PER_SLICE_ROW, _PER_SLICE_BOTH) for c in slice_classes) and any(
        c == _PER_SLICE_ROW for c in slice_classes
    ):
        return _ROW_STOCHASTIC
    return "other"


def _candidate_readings(values: Any, shape: list[int]) -> dict[str, list[Matrix]]:
    """Per-action slice matrices under each candidate action-axis position.

    ``inner``: canonical declaration order ``(next, prev, action)`` — slice
    ``a`` is ``values[:,:,a]`` with rows = next states, columns = previous
    states. ``outer``: each outer element of the literal is one per-action
    slice (action-outer storage). A 2-D literal has a single reading (its
    one matrix) and no action axis. Readings whose elements are not
    rectangular numeric matrices are omitted.
    """
    if len(shape) == 2:
        matrix = numeric_matrix(values)
        if matrix is None:
            return {}
        return {_AXIS_INNER: [matrix]}
    if len(shape) != 3:
        return {}
    readings: dict[str, list[Matrix]] = {}
    inner: MatrixStack = []
    for a in range(shape[2]):
        rows = [[values[n][p][a] for p in range(shape[1])] for n in range(shape[0])]
        matrix = numeric_matrix(rows)
        if matrix is None:
            inner = []
            break
        inner.append(matrix)
    if inner:
        readings[_AXIS_INNER] = inner
    outer: MatrixStack = []
    for element in values:
        matrix = numeric_matrix(element)
        if matrix is None:
            outer = []
            break
        outer.append(matrix)
    if outer:
        readings[_AXIS_OUTER] = outer
    return readings


def transpose_b_to_canonical(values: list[Any], action_axis: str | None) -> list[Any]:
    """Return the canonical ``(next, prev, action)``-nested copy of ``values``.

    Matches the transposition semantics of ``canonicalize_pomdp`` in
    ``gnn.extract.pomdp_extractor`` for the action-outer textbook layout:
    ``canonical[n][p][a] = stored[a][p][n]``. A row-stochastic literal
    nested ``(prev, next, action)`` (action innermost) swaps its first two
    axes; a 2-D row-stochastic matrix is transposed in place. The input is
    never mutated.
    """
    shape = _nested_shape(values)
    if action_axis == _AXIS_OUTER and len(shape) == 3:
        # (action, prev, next) -> (next, prev, action)
        return [
            [[values[a][p][n] for a in range(shape[0])] for p in range(shape[1])]
            for n in range(shape[2])
        ]
    if action_axis == _AXIS_INNER and len(shape) == 3:
        # (prev, next, action) -> (next, prev, action)
        return [
            [[values[p][n][a] for p in range(shape[0])] for n in range(shape[1])]
            for a in range(shape[2])
        ]
    if len(shape) == 2:
        # rows <-> columns
        return [[values[p][n] for p in range(shape[0])] for n in range(shape[1])]
    raise ValueError(
        f"cannot transpose B literal with shape {shape} and action axis {action_axis!r}"
    )


def _state_factor_description(
    dimensions: dict[str, Any], state_size: int | None
) -> str:
    """Describe B's state factor, naming a unique ``s*`` variable when one matches."""
    if state_size is None:
        return "state factor"
    candidates = [
        name
        for name, dims in dimensions.items()
        if isinstance(dims, list)
        and bool(dims)
        and dims[0] == state_size
        and re.fullmatch(r"s(?:_[A-Za-z0-9]+)?", name) is not None
        and "prime" not in name
    ]
    if len(candidates) == 1:
        return f"state factor {candidates[0]} ({state_size} states)"
    return f"state factor ({state_size} states)"


def _format_slice_list(indices: list[int]) -> str:
    """Render flipped action-slice indices for warning text."""
    if len(indices) == 1:
        return f"action slice {indices[0]}"
    return f"action slices [{', '.join(str(i) for i in indices)}]"


def _state_size(dims: list[int], shape: list[int]) -> int | None:
    """Previous-state axis size (canonical order: second axis)."""
    if len(dims) >= 2:
        return dims[1]
    if len(dims) == 1:
        return dims[0]
    if len(shape) >= 2:
        return shape[1]
    return None


def _scan_tensor(values: Any, dims: list[int], *, transpose_b: bool) -> dict[str, Any]:
    """Classify one B tensor literal and optionally transpose it canonically."""
    shape = _nested_shape(values)
    readings = _candidate_readings(values, shape)
    slice_classes = {
        axis: [_classify_slice(matrix) for matrix in classes]
        for axis, classes in readings.items()
    }
    kinds = {axis: _reading_kind(classes) for axis, classes in slice_classes.items()}

    def _preferred(axis_pool: list[str]) -> str:
        """Canonical declaration order (action innermost) wins ties."""
        if _AXIS_INNER in axis_pool:
            return _AXIS_INNER
        return axis_pool[0]

    canonical_axes = sorted(axis for axis, kind in kinds.items() if kind == _CANONICAL)
    ambiguous_axes = sorted(axis for axis, kind in kinds.items() if kind == _AMBIGUOUS)
    row_axes = sorted(axis for axis, kind in kinds.items() if kind == _ROW_STOCHASTIC)

    orientation = _NON_STOCHASTIC
    action_axis: str | None = None
    flipped: list[int] = []
    if canonical_axes:
        orientation = _CANONICAL
        action_axis = _preferred(canonical_axes)
    elif ambiguous_axes:
        orientation = _AMBIGUOUS
        action_axis = _preferred(ambiguous_axes)
    elif row_axes:
        orientation = _ROW_STOCHASTIC
        action_axis = _preferred(row_axes)
        flipped = [
            index
            for index, cls in enumerate(slice_classes.get(action_axis, []))
            if cls == _PER_SLICE_ROW
        ]
    if len(shape) == 2:
        # A 2-D transition matrix has no action axis; orientation only.
        action_axis = None

    tensor: dict[str, Any] = {
        "variable": "B",
        "dims": dims or shape,
        "action_axis": action_axis,
        "orientation": orientation,
        "flipped_slices": flipped,
        "transposed": False,
        "previous_orientation": None,
        "canonical_after_transpose": None,
    }

    if transpose_b and orientation == _ROW_STOCHASTIC:
        transposed = transpose_b_to_canonical(values, action_axis)
        transposed_shape = _nested_shape(transposed)
        tensor["transposed"] = True
        tensor["previous_orientation"] = _ROW_STOCHASTIC
        tensor["canonical_after_transpose"] = any(
            _reading_kind([_classify_slice(matrix) for matrix in slices]) == _CANONICAL
            for slices in _candidate_readings(transposed, transposed_shape).values()
        )
    return tensor


def scan_b_orientation(content: str, *, transpose_b: bool = False) -> dict[str, Any]:
    """Run the B-orientation diagnostic over raw GNN content text.

    Returns a receipt with per-tensor orientation findings, warnings for
    textbook (row-stochastic) orientations, informational notes for
    orientation-ambiguous (doubly stochastic) tensors, and — with
    ``transpose_b`` — the canonical transposition recorded per tensor.
    Never raises.
    """
    warnings: list[str] = []
    notes: list[str] = []
    tensors: list[dict[str, Any]] = []
    try:
        evidence = extract_b_matrix_evidence(content)
        values = evidence.get("values")
        if values is not None:
            dimensions = extract_gnn_dimensions(content)
            dims_raw = dimensions.get("B")
            dims = [int(d) for d in dims_raw] if isinstance(dims_raw, list) else []
            tensor = _scan_tensor(values, dims, transpose_b=transpose_b)
            tensors.append(tensor)
            var_dims = tensor["dims"]
            dims_text = ",".join(str(d) for d in var_dims)
            state_desc = _state_factor_description(
                dimensions, _state_size(dims, _nested_shape(values))
            )
            if tensor["orientation"] == _ROW_STOCHASTIC:
                warnings.append(
                    f"[B-orientation] Transition tensor B[{dims_text}] "
                    f"({state_desc}): "
                    f"{_format_slice_list(tensor['flipped_slices'])} "
                    "are row-stochastic — rows sum to 1 over next states, the "
                    "textbook POMDP layout (rows = previous state s_t, columns "
                    "= next state s_{t+1}). Canonical GNN order is "
                    "B[next_state, previous_state, action] with "
                    "column-stochastic slices (rows = next states, each column "
                    "= one previous state), so canonical readers silently read "
                    "this tensor transposed and every transition probability "
                    "flips. Fix: transpose each per-action slice (rows <-> "
                    "columns) and store the action axis last, or run Step 6 "
                    "with --transpose-b to validate the canonical "
                    "transposition. Convention: docs/gnn/gnn_syntax.md "
                    "(B-tensor orientation)."
                )
                if tensor["transposed"]:
                    outcome = (
                        "canonical"
                        if tensor["canonical_after_transpose"]
                        else "still not canonical"
                    )
                    notes.append(
                        f"[B-orientation] B[{dims_text}]: transposed to "
                        f"canonical order; previous orientation was "
                        f"{_ROW_STOCHASTIC} ({outcome})."
                    )
            elif tensor["orientation"] == _AMBIGUOUS:
                notes.append(
                    f"[B-orientation] B[{dims_text}]: per-action slices are "
                    "doubly stochastic (rows and columns both sum to 1), so "
                    "the row/column orientation cannot be determined from the "
                    "data; declaration comments decide the reading. Canonical "
                    "order B[next_state, previous_state, action] expects "
                    "column-stochastic slices."
                )
    except Exception as error:  # best-effort diagnostic, never fatal
        logger.debug("B-orientation scan failed: %s", error)
        return {
            "valid": False,
            "warnings": warnings,
            "notes": notes,
            "tensors": tensors,
            "orientation_score": 0.0,
            "recovery": True,
            "error": str(error),
        }
    return {
        "valid": True,
        "warnings": warnings,
        "notes": notes,
        "tensors": tensors,
        "orientation_score": clamp01(1.0 - 0.05 * len(warnings)),
        "recovery": False,
    }


def check_b_orientation(
    model_data: str | Path | ModelData,
    *,
    transpose_b: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Step-6 stage: B-tensor orientation diagnostic for one model.

    Accepts the same inputs as the sibling stages (path or parsed-model
    mapping) and returns a stage receipt with ``warnings`` (textbook
    orientation), ``notes`` (ambiguous / transposition records), per-tensor
    ``tensors`` findings, and an ``orientation_score``. Warnings are
    advisory and never invalidate the file; the existing stochasticity
    error paths keep owning non-stochastic tensors. Never raises.
    """
    try:
        if isinstance(model_data, Mapping):
            file_path = str(model_data.get("file_path", "unknown"))
            content = extract_content_from_dict(model_data)
        elif isinstance(model_data, (str, Path)):
            path = Path(model_data)
            file_path = str(path)
            content = path.read_text(encoding="utf-8")
        else:
            raise TypeError(
                f"model_data must be a path or mapping, got {type(model_data).__name__}"
            )
        result = scan_b_orientation(content, transpose_b=transpose_b)
        result.update(
            file_path=file_path,
            file_name=display_file_name(file_path),
        )
        return result
    except Exception as error:
        file_path = (
            str(model_data) if isinstance(model_data, (str, Path)) else "unknown"
        )
        return {
            "status": "error",
            "file_path": file_path,
            "file_name": display_file_name(file_path),
            "error": str(error),
            "valid": False,
            "warnings": [str(error)],
            "notes": [],
            "tensors": [],
            "orientation_score": 0.0,
            "recovery": True,
        }
