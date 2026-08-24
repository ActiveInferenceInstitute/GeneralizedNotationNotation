"""
Dimension extraction and compatibility validation for GNN models.

Provides functions to parse the ``StateSpaceBlock`` from GNN file content
and validate Active Inference POMDP dimensional constraints (A, B, C, D).
"""

from __future__ import annotations

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


def validate_dimension_compatibility(variables: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that matrix/tensor dimensions are compatible in a GNN model.

    Checks Active Inference POMDP constraints:

    - Likelihood matrix ``A[obs, states]``: columns must match hidden state count
    - Transition tensor ``B[states, states, actions]``: first two dims must match
    - Preference ``C[obs]``: length must match A's first dimension
    - Prior ``D[states]``: length must match A's second dimension

    Args:
        variables: Dict mapping variable names to their dimension specs,
                   e.g. ``{"A": [3,3], "B": [3,3,3], "s": [3,1]}``.

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
