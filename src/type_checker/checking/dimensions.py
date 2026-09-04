"""
Dimension extraction and compatibility validation for GNN models.

Provides functions to parse the ``StateSpaceBlock`` from GNN file content
and validate Active Inference POMDP dimensional constraints (A, B, C, D).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from math import prod
from typing import Any, Dict

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedVariable:
    """State-space declaration needed by type and resource analysis."""

    name: str
    dimensions: tuple[str, ...]
    dtype: str
    line: int


_VARIABLE_DECLARATION = re.compile(
    r"^(?P<name>[^\s\[\],()><|]+)\s*\[(?P<body>[^\]]+)\]\s*(?:#.*)?$"
)


def _normalize_identifier(name: str) -> str:
    return "π" if name.strip().lower() == "pi" else name.strip()


def _append_once(items: list[str], message: str) -> None:
    """Append a diagnostic once while preserving discovery order."""
    if message not in items:
        items.append(message)


def _state_space_lines(content: str) -> tuple[bool, list[tuple[int, str]]]:
    """Return whether the section exists and its meaningful numbered lines."""
    found_section = False
    in_section = False
    lines: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            in_section = stripped[3:].strip() == "StateSpaceBlock"
            found_section = found_section or in_section
            continue
        if in_section and stripped and not stripped.startswith("#"):
            lines.append((line_number, stripped))
    return found_section, lines


def parse_state_variables(content: str) -> tuple[list[ParsedVariable], list[str]]:
    """Parse canonical state declarations, retaining symbolic dimensions."""
    found_section, lines = _state_space_lines(content)
    variables: list[ParsedVariable] = []
    diagnostics: list[str] = []
    seen: set[str] = set()

    for line_number, line in lines:
        match = _VARIABLE_DECLARATION.fullmatch(line)
        if match is None:
            diagnostics.append(
                f"Unparseable StateSpaceBlock declaration at line {line_number}: '{line}'"
            )
            continue

        name = _normalize_identifier(match.group("name"))
        dimensions: list[str] = []
        dtype = "float"
        for part in match.group("body").split(","):
            token = part.strip()
            if not token:
                continue
            if "=" in token:
                key, value = (item.strip() for item in token.split("=", 1))
                if key == "type":
                    dtype = value
                continue
            dimensions.append(_normalize_identifier(token))

        if name in seen:
            diagnostics.append(
                f"[GNN-E004] Duplicate variable declaration: '{name}' at line {line_number}"
            )
        seen.add(name)
        variables.append(
            ParsedVariable(
                name=name,
                dimensions=tuple(dimensions),
                dtype=dtype,
                line=line_number,
            )
        )

    if not found_section:
        diagnostics.append("Missing StateSpaceBlock section")
    elif not variables:
        diagnostics.append("StateSpaceBlock contains no valid variable declarations")
    return variables, diagnostics


def _resolve_dimensions(
    variable: ParsedVariable,
    variables: dict[str, ParsedVariable],
    cache: dict[str, list[int]],
    diagnostics: list[str],
    resolving: tuple[str, ...] = (),
) -> list[int]:
    """Resolve numeric and variable-backed symbolic dimensions safely."""
    if variable.name in cache:
        return cache[variable.name]
    if variable.name in resolving:
        cycle = " -> ".join((*resolving, variable.name))
        _append_once(diagnostics, f"Cyclic symbolic dimension reference: {cycle}")
        return [1]

    if not variable.dimensions:
        _append_once(
            diagnostics,
            f"Variable '{variable.name}' has no dimensions; using a scalar fallback",
        )
        cache[variable.name] = [1]
        return cache[variable.name]

    resolved: list[int] = []
    next_resolving = (*resolving, variable.name)
    for raw_dimension in variable.dimensions:
        token = str(raw_dimension).strip()
        try:
            dimension = int(token)
        except ValueError:
            referenced = variables.get(token)
            if referenced is None:
                _append_once(
                    diagnostics,
                    f"Variable '{variable.name}' has unresolved dimension '{token}'; "
                    "using 1 for estimation",
                )
                resolved.append(1)
                continue
            referenced_dimensions = _resolve_dimensions(
                referenced,
                variables,
                cache,
                diagnostics,
                next_resolving,
            )
            resolved.append(prod(referenced_dimensions))
            continue

        if dimension <= 0:
            _append_once(
                diagnostics,
                f"Variable '{variable.name}' has non-positive dimension {dimension}; "
                "using 1 for estimation",
            )
            resolved.append(1)
        else:
            resolved.append(dimension)

    cache[variable.name] = resolved
    return resolved


def extract_gnn_dimensions_with_diagnostics(
    content: str,
) -> tuple[dict[str, list[int]], list[str]]:
    """Extract resolved dimensions and explicit diagnostics from GNN content."""
    parsed_variables, diagnostics = parse_state_variables(content)

    variables: dict[str, ParsedVariable] = {}
    for variable in parsed_variables:
        variables.setdefault(variable.name, variable)

    cache: dict[str, list[int]] = {}
    dimensions = {
        name: _resolve_dimensions(variable, variables, cache, diagnostics)
        for name, variable in variables.items()
    }
    return dimensions, diagnostics


def extract_gnn_dimensions(content: str) -> Dict[str, Any]:
    """Extract variable dimensions from GNN StateSpaceBlock content.

    Parses patterns like: ``A[3,3,type=float]``, ``s[3,1,type=float]``.

    Args:
        content: Full GNN file content as string.

    Returns:
        Dict mapping variable names to their dimension lists.
    """
    dimensions, diagnostics = extract_gnn_dimensions_with_diagnostics(content)
    for diagnostic in diagnostics:
        _logger.log(5, diagnostic)
    return dimensions


_ROW_PREV_CONVENTION = re.compile(
    r"rows?\s+(?:are|index)\s+(?:the\s+)?previous", re.IGNORECASE
)
_COL_NEXT_CONVENTION = re.compile(
    r"columns?\s+(?:are|index)\s+(?:the\s+)?next", re.IGNORECASE
)
_ROW_NEXT_CONVENTION = re.compile(
    r"rows?\s+(?:are|index)\s+(?:the\s+)?next", re.IGNORECASE
)
_COL_PREV_CONVENTION = re.compile(
    r"columns?\s+(?:are|index)\s+(?:the\s+)?previous", re.IGNORECASE
)
_COL_ACTION_CONVENTION = re.compile(
    r"columns?\s+(?:are|index)\s+(?:the\s+)?actions?\b", re.IGNORECASE
)
_STOCHASTICITY_TOLERANCE = 1e-6


def _declaration_orientation(comment: str) -> str | None:
    """Classify a B declaration comment as canonical, old, or unspecified.

    Canonical means the declared axis order starts with the next-state axis
    (``B[next_state, previous_state, action]``); old means the previous-state
    axis is declared first.
    """
    if not comment:
        return None
    lowered = comment.lower().replace("_", " ")
    idx_next = lowered.find("next")
    idx_prev = lowered.find("prev")
    if idx_next == -1 or idx_prev == -1:
        return None
    return "canonical" if idx_next < idx_prev else "old"


def _comment_orientation(comment: str) -> str | None:
    """Classify an InitialParameterization B comment's claimed slice layout.

    Old orientation: per-action slices claiming ``rows are previous states,
    columns are next states``. Canonical claims: per-action slices with
    ``rows are next states, columns are previous states``, or next-state-
    outer slices (the GridWorld nesting) with ``rows are previous states,
    columns are actions``.
    """
    has_row_prev = _ROW_PREV_CONVENTION.search(comment) is not None
    has_row_next = _ROW_NEXT_CONVENTION.search(comment) is not None
    has_col_next = _COL_NEXT_CONVENTION.search(comment) is not None
    has_col_prev = _COL_PREV_CONVENTION.search(comment) is not None
    has_col_action = _COL_ACTION_CONVENTION.search(comment) is not None
    claims_old = has_row_prev and has_col_next
    claims_canonical = (has_row_next and has_col_prev) or (
        has_row_prev and has_col_action
    )
    if claims_old and not claims_canonical:
        return "old"
    if claims_canonical and not claims_old:
        return "canonical"
    return None


def _numeric_matrix(rows: Any) -> list[list[float]] | None:
    """Return a rectangular float matrix, or None when the literal is not one."""
    if not isinstance(rows, list) or not rows:
        return None
    matrix: list[list[float]] = []
    width: int | None = None
    for row in rows:
        if not isinstance(row, list):
            return None
        values: list[float] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                return None
            values.append(float(cell))
        if width is None:
            width = len(values)
        elif len(values) != width:
            return None
        matrix.append(values)
    return matrix


def _all_close_one(sums: list[float]) -> bool:
    """Return True when every sum is within tolerance of 1.0."""
    return all(abs(total - 1.0) <= _STOCHASTICITY_TOLERANCE for total in sums)


def _b_orientation_verdict(values: Any) -> str:
    """Classify nested B values by stochasticity of their per-action slices.

    Returns ``"doubly"`` (every slice row- and column-stochastic),
    ``"column"`` (column-stochastic only — canonical orientation),
    ``"row"`` (row-stochastic only — old orientation under canonical
    reading), ``"cross"`` (sums over the outer axis equal one per cell,
    the GridWorld next-state-outer nesting), ``"neither"``, or
    ``"invalid"`` (not a numeric nested matrix).
    """
    if (
        isinstance(values, list)
        and values
        and isinstance(values[0], list)
        and values[0]
        and isinstance(values[0][0], list)
    ):
        raw_slices: list[Any] = list(values)
    elif isinstance(values, list) and values and isinstance(values[0], list):
        raw_slices = [values]
    else:
        return "invalid"

    matrices: list[list[list[float]]] = []
    for raw_slice in raw_slices:
        matrix = _numeric_matrix(raw_slice)
        if matrix is None:
            return "invalid"
        matrices.append(matrix)

    row_stochastic = True
    column_stochastic = True
    for matrix in matrices:
        row_sums = [sum(row) for row in matrix]
        col_sums = [
            sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))
        ]
        if not _all_close_one(row_sums):
            row_stochastic = False
        if not _all_close_one(col_sums):
            column_stochastic = False
    if row_stochastic and column_stochastic:
        return "doubly"
    if column_stochastic:
        return "column"

    cross_consistent = False
    if len(matrices) > 1 and len({(len(m), len(m[0])) for m in matrices}) == 1:
        cross = [
            [sum(m[i][j] for m in matrices) for j in range(len(matrices[0][0]))]
            for i in range(len(matrices[0]))
        ]
        cross_consistent = _all_close_one([total for row in cross for total in row])
    if row_stochastic:
        # Degenerate models (e.g. prev-state-independent transitions) can be
        # row-stochastic per slice while still summing to 1 over the outer
        # (next_state) axis — canonically explainable, so only flag the data
        # when no canonical reading exists.
        return "row" if not cross_consistent else "cross"
    return "cross" if cross_consistent else "neither"


def _collect_braced_block(lines: list[str], start_idx: int) -> str | None:
    """Return the balanced ``{...}`` block beginning at ``lines[start_idx]``."""
    joined: list[str] = []
    depth = 0
    opened = False
    for line in lines[start_idx:]:
        for char in line:
            if char == "{":
                depth += 1
                opened = True
            elif char == "}":
                depth -= 1
        joined.append(line)
        if opened and depth <= 0:
            return "\n".join(joined)
    return None


def _parse_braced_nested_literal(block: str) -> Any:
    """Parse a ``B={ ... }`` block body into nested Python lists."""
    brace_start = block.find("{")
    if brace_start == -1:
        return None
    inner = block[brace_start + 1 : block.rindex("}")]
    cleaned = inner.replace("(", "[").replace(")", "]")
    cleaned = re.sub(r",\s*(\])", r"\1", cleaned)
    try:
        return ast.literal_eval("[" + cleaned + "]")
    except (ValueError, SyntaxError):
        return None


def _validate_b_orientation(
    dims: Dict[str, list[int]],
    b_evidence: Dict[str, Any],
    strict: bool,
    issues: list[str],
    warnings: list[str],
) -> None:
    """Flag positive B orientation contradictions per the canonical order.

    Canonical order is ``B[next_state, previous_state, action]`` = pymdp
    1.0.0 ``B[s',s,a]``: per-action slices are written rows = next states,
    columns = previous states, column-stochastic over next_state.

    Errors (strict) / warnings (non-strict) are raised only for:
      * comment-vs-comment contradiction (declaration vs parameterization),
      * row-stochastic-only slices under the canonical reading.
    Doubly-stochastic (orientation-ambiguous) data passes.
    """
    values = b_evidence.get("values")
    if "B" not in dims or values is None:
        return
    verdict = _b_orientation_verdict(values)

    declaration_orientation = _declaration_orientation(
        str(b_evidence.get("declaration_comment") or "")
    )
    parameterization_orientation = _comment_orientation(
        str(b_evidence.get("parameterization_comment") or "")
    )
    target = issues if strict else warnings
    if (
        declaration_orientation is not None
        and parameterization_orientation is not None
        and declaration_orientation != parameterization_orientation
    ):
        target.append(
            "[GNN-E002] Transition matrix B: contradictory axis-order comments. "
            f"The StateSpaceBlock declaration claims a {declaration_orientation} "
            f"per-slice layout while the InitialParameterization comment claims "
            f"{parameterization_orientation}. Canonical order is "
            "B[next_state, previous_state, action] = pymdp 1.0.0 B[s',s,a]; "
            "per-action slices are written rows = next states, columns = "
            "previous states, column-stochastic over next_state. Align the "
            "comments to the canonical order."
        )
    if verdict == "row":
        target.append(
            "[GNN-E002] Transition matrix B: per-action slices are "
            "row-stochastic only (rows sum to 1, columns do not). Under the "
            "canonical reading B[next_state, previous_state, action] each "
            "slice must be column-stochastic: rows are next states, columns "
            "are previous states, and each column (one previous state) sums "
            "to 1 over next states. Transpose each slice (rows <-> cols)."
        )


def extract_b_matrix_evidence(content: str) -> Dict[str, Any]:
    """Extract B matrix values plus orientation comments from GNN content.

    Returns a dict with ``values`` (nested numeric lists parsed from the
    ``InitialParameterization`` ``B={...}`` block, or ``None`` when absent
    or unparseable), ``declaration_comment`` (the comment attached to the
    ``B[...]`` StateSpaceBlock line), and ``parameterization_comment``
    (the comment block preceding ``B=``).
    """
    evidence: Dict[str, Any] = {
        "values": None,
        "declaration_comment": "",
        "parameterization_comment": "",
    }
    lines = content.splitlines()
    section = ""
    pending: list[str] = []
    for line_number, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip()
            pending = []
            continue
        if not stripped:
            continue
        if stripped.startswith("#"):
            pending.append(stripped[1:].strip())
            continue
        if section == "StateSpaceBlock" and stripped.startswith("B["):
            inline = stripped.split("#", 1)
            if len(inline) == 2:
                pending.append(inline[1].strip())
            evidence["declaration_comment"] = " ".join(part for part in pending if part)
        elif section == "InitialParameterization" and re.match(r"^B\s*=", stripped):
            evidence["parameterization_comment"] = " ".join(
                part for part in pending if part
            )
            if "{" in stripped:
                block = _collect_braced_block(lines, line_number)
                if block is not None:
                    evidence["values"] = _parse_braced_nested_literal(block)
        pending = []
    return evidence


def validate_dimension_compatibility(
    variables: Dict[str, Any],
    *,
    b_evidence: Dict[str, Any] | None = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate that matrix/tensor dimensions are compatible in a GNN model.

    Checks Active Inference POMDP constraints:

    - Likelihood matrix ``A[obs, states]``: columns must match hidden state count
    - Transition tensor ``B[states, states, actions]``: first two dims must match
    - Preference ``C[obs]``: length must match A's first dimension
    - Prior ``D[states]``: length must match A's second dimension

    Args:
        variables: Dict mapping variable names to their dimension specs,
                   e.g. ``{"A": [3,3], "B": [3,3,3], "s": [3,1]}``.
        b_evidence: Optional output of :func:`extract_b_matrix_evidence`
                    (B matrix values plus the two orientation comments).
                    When provided, B orientation/contradiction checks run
                    on the matrix values and comments.
        strict: When True, orientation contradictions are errors (in
                ``issues``); otherwise they are warnings.

    Returns:
        Dict with keys: ``compatible`` (bool), ``issues`` (list), ``warnings`` (list),
        ``variables_checked`` (list), ``dimension_map`` (dict).
    """
    issues: list[str] = []
    warnings: list[str] = []

    # Parse dimension specs: extract variables with numeric dimensions
    dims: Dict[str, list[int]] = {}
    for name, spec in variables.items():
        if isinstance(spec, (list, tuple)) and all(isinstance(d, int) for d in spec):
            dims[name] = list(spec)

    # Check A-s compatibility: A[obs, states], s[states, 1]
    if "A" in dims and "s" in dims:
        a_dims = dims["A"]
        s_dims = dims["s"]
        if len(a_dims) >= 2 and len(s_dims) >= 1:
            if a_dims[1] != s_dims[0]:
                issues.append(
                    f"Dimension mismatch: A[{a_dims[0]},{a_dims[1]}] column count ({a_dims[1]}) "
                    f"!= s[{s_dims[0]},...] row count ({s_dims[0]}). "
                    f"A's columns must equal the number of hidden states."
                )

    # Check B symmetry: B[states, states, actions] -- first two dims must match
    if "B" in dims:
        b_dims = dims["B"]
        if len(b_dims) >= 2 and b_dims[0] != b_dims[1]:
            issues.append(
                f"Transition matrix B[{','.join(str(d) for d in b_dims)}]: "
                f"first two dimensions must match (got {b_dims[0]} != {b_dims[1]}). "
                f"B[next_states, prev_states, actions] requires next_states == prev_states."
            )

    # Check A-B state dimension consistency
    if "A" in dims and "B" in dims:
        a_dims = dims["A"]
        b_dims = dims["B"]
        if len(a_dims) >= 2 and len(b_dims) >= 1:
            if a_dims[1] != b_dims[0]:
                issues.append(
                    f"State dimension mismatch between A and B: "
                    f"A has {a_dims[1]} hidden states, B has {b_dims[0]} states. "
                    f"Must be equal."
                )

    # Check C-A observation compatibility
    if "C" in dims and "A" in dims:
        c_dims = dims["C"]
        a_dims = dims["A"]
        if len(c_dims) >= 1 and len(a_dims) >= 1:
            c_obs = c_dims[0]
            a_obs = a_dims[0]
            if c_obs != a_obs:
                issues.append(
                    f"Preference vector C[{c_obs}] length != A observation dimension A[{a_obs},...]. "
                    f"C must have one entry per observation outcome."
                )

    # Check D-s prior compatibility
    if "D" in dims and "s" in dims:
        d_dims = dims["D"]
        s_dims = dims["s"]
        if len(d_dims) >= 1 and len(s_dims) >= 1:
            if d_dims[0] != s_dims[0]:
                issues.append(
                    f"Prior D[{d_dims[0]}] length != hidden state s[{s_dims[0]},...] count. "
                    f"D must have one entry per hidden state."
                )

    # Check B orientation/contradiction when matrix values + comments are
    # supplied (b_evidence). Only positive contradictions are flagged.
    if b_evidence is not None:
        _validate_b_orientation(dims, b_evidence, strict, issues, warnings)

    # Warn about very large dimensions (tractability)
    for name, d in dims.items():
        total_elements = 1
        for dim in d:
            total_elements *= dim
        if total_elements > 10000:
            warnings.append(
                f"Variable {name} with dimensions {d} has {total_elements} total elements. "
                f"Consider dimensionality reduction for tractable inference."
            )

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "variables_checked": list(dims.keys()),
        "dimension_map": dims,
    }
