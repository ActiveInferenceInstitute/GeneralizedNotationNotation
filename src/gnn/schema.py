#!/usr/bin/env python3
"""
GNN Schema Module  — JSON Schema definitions and validation for parsed GNN objects.

Provides:
  - GNNParseError: structured error with code, line number, and message
  - GNN_MODEL_SCHEMA: JSON Schema dict for a single parsed GNN model
  - validate_gnn_object(): validates a dict against the schema
  - parse_connections(): extracts edges with optional annotations
  - validate_matrix_dimensions(): cross-checks parameterization vs declarations
"""

import ast
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── GNN Error Types ────────────────────────────────────────────────────────────


@dataclass
class GNNParseError:
    """Structured parse error with code, line number, and human-readable message."""

    code: str  # e.g. GNN-E001
    message: str
    line: Optional[int] = None
    file: Optional[str] = None
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        """Return the string representation."""
        loc = f":{self.line}" if self.line else ""
        src = f" [{self.file}{loc}]" if self.file else ""
        return f"[{self.code}] {self.message}{src}"


# ─── Connection Parsing ─────────────────────────────────────────────────────────

# Matches: A>B, A-B, A>B:label, A-B:label
_CONNECTION_RE = re.compile(
    r"^(?P<source>[A-Za-z_π'][A-Za-z0-9_π']*)"
    r"(?P<op>[>\-])"
    r"(?P<target>[A-Za-z_π'][A-Za-z0-9_π']*)"
    r"(?::(?P<label>[A-Za-z0-9_]+))?$"
)


@dataclass
class GNNConnectionEdge:
    """A parsed connection/edge between two state-space variables."""

    source: str
    target: str
    directed: bool  # True for '>', False for '-'
    label: Optional[str] = None
    line: Optional[int] = None


def parse_connections(
    content: str,
    *,
    known_variables: Optional[set] = None,
    file_path: Optional[str] = None,
) -> Tuple[List[GNNConnectionEdge], List[GNNParseError]]:
    """
    Parse the Connections section of a GNN file.

    Args:
        content: Full GNN file content (or just the Connections section).
        known_variables: Set of declared variable names for cross-validation.
        file_path: Optional source file for error reporting.

    Returns:
        (connections, errors) tuple.
    """
    connections: List[GNNConnectionEdge] = []
    errors: List[GNNParseError] = []

    # Find the ## Connections section
    in_connections = False
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        # Detect section boundaries
        if line.startswith("## "):
            section_name = line[3:].strip()
            in_connections = section_name == "Connections"
            continue

        if not in_connections:
            continue
        if not line or line.startswith("#"):
            continue

        m = _CONNECTION_RE.match(line)
        if m:
            conn = GNNConnectionEdge(
                source=m.group("source"),
                target=m.group("target"),
                directed=(m.group("op") == ">"),
                label=m.group("label"),
                line=line_no,
            )
            connections.append(conn)

            # Cross-validate against declared variables
            if known_variables is not None:
                if conn.source not in known_variables:
                    errors.append(
                        GNNParseError(
                            code="GNN-W002",
                            message=f"Connection references undeclared variable '{conn.source}'",
                            line=line_no,
                            file=file_path,
                            severity="warning",
                        )
                    )
                if conn.target not in known_variables:
                    errors.append(
                        GNNParseError(
                            code="GNN-W002",
                            message=f"Connection references undeclared variable '{conn.target}'",
                            line=line_no,
                            file=file_path,
                            severity="warning",
                        )
                    )
        else:
            errors.append(
                GNNParseError(
                    code="GNN-E005",
                    message=f"Unparseable connection: '{line}'",
                    line=line_no,
                    file=file_path,
                )
            )

    return connections, errors


# ─── Variable Parsing (enhanced) ────────────────────────────────────────────────

_VAR_RE = re.compile(
    r"^(?P<name>[A-Za-z_π'][A-Za-z0-9_π']*)"
    r"\[(?P<dims>[^\]]+)\]"
)


@dataclass
class GNNVariable:
    """A parsed state-space variable declaration."""

    name: str
    dimensions: List[str]  # can be ints or symbolic names
    dtype: str = "float"
    default: Optional[str] = None
    line: Optional[int] = None


def parse_state_space(
    content: str,
    *,
    file_path: Optional[str] = None,
) -> Tuple[List[GNNVariable], List[GNNParseError]]:
    """
    Parse the StateSpaceBlock section of a GNN file.

    Returns:
        (variables, errors) tuple.
    """
    variables: List[GNNVariable] = []
    errors: List[GNNParseError] = []
    seen_names: set = set()

    in_ssb = False
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        if line.startswith("## "):
            section_name = line[3:].strip()
            in_ssb = section_name == "StateSpaceBlock"
            continue

        if not in_ssb:
            continue
        if not line or line.startswith("#"):
            continue

        # Strip inline comment
        if "#" in line:
            line = line[: line.index("#")].strip()

        m = _VAR_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        raw_dims = m.group("dims")

        # Parse comma-separated dimension entries
        parts = [p.strip() for p in raw_dims.split(",")]
        dims: list[Any] = []
        dtype = "float"
        default = None
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                key, val = key.strip(), val.strip()
                if key == "type":
                    dtype = val
                elif key == "default":
                    default = val
            else:
                dims.append(part)

        if name in seen_names:
            errors.append(
                GNNParseError(
                    code="GNN-E004",
                    message=f"Duplicate variable declaration: '{name}'",
                    line=line_no,
                    file=file_path,
                )
            )
        seen_names.add(name)

        variables.append(
            GNNVariable(
                name=name,
                dimensions=dims,
                dtype=dtype,
                default=default,
                line=line_no,
            )
        )

    return variables, errors


# ─── Matrix Dimension Validation ─────────────────────────────────────────────────


_PARAM_ASSIGN_RE = re.compile(
    r"^(?P<name>[A-Za-z_π'][A-Za-z0-9_π']*)\s*=\s*(?P<value>.*)$"
)


def _strip_parameter_comment(line: str) -> str:
    """Remove a GNN inline comment from one parameterization line."""
    if "#" not in line:
        return line.strip()
    return line[: line.index("#")].strip()


def _parse_parameter_value(raw_value: str) -> Any:
    """Parse GNN tuple/braced parameter syntax as a Python literal structure."""
    literal_text = raw_value.strip().replace("{", "[").replace("}", "]")
    return ast.literal_eval(literal_text)


def _parameter_shape(value: Any) -> tuple[list[int], bool]:
    """Return the nested sequence shape and whether any nested row is ragged."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return [0], False
        child_shapes: list[list[int]] = []
        ragged = False
        for child in value:
            child_shape, child_ragged = _parameter_shape(child)
            child_shapes.append(child_shape)
            ragged = ragged or child_ragged
        first_shape = child_shapes[0]
        if any(shape != first_shape for shape in child_shapes[1:]):
            ragged = True
        return [len(value), *first_shape], ragged
    return [], False


def _numeric_dimensions(dimensions: list[str]) -> list[int] | None:
    """Return numeric dimensions or None when any dimension is symbolic."""
    try:
        return [int(dim) for dim in dimensions]
    except (TypeError, ValueError):
        return None


def _shape_for_comparison(
    actual_shape: list[int], expected_shape: list[int]
) -> list[int]:
    """Normalize established GNN parameter layout conventions for comparison."""
    if len(expected_shape) == 1 and len(actual_shape) == 2 and actual_shape[0] == 1:
        return actual_shape[1:]
    if len(expected_shape) == 2 and expected_shape[1] == 1:
        if len(actual_shape) == 1 and actual_shape[0] == expected_shape[0]:
            return expected_shape
        if len(actual_shape) == 2 and actual_shape == [1, expected_shape[0]]:
            return expected_shape
    if len(expected_shape) == 3 and len(actual_shape) == 3:
        action_major_shape = [expected_shape[2], expected_shape[0], expected_shape[1]]
        if actual_shape == action_major_shape:
            return expected_shape
    return actual_shape


def _format_shape(shape: list[int]) -> str:
    """Format a numeric shape for diagnostics."""
    return "scalar" if not shape else "x".join(str(part) for part in shape)


def _validate_parameter_assignment(
    *,
    variable_name: str,
    raw_value: str,
    end_line: int,
    var_map: dict[str, GNNVariable],
    file_path: Optional[str],
) -> list[GNNParseError]:
    """Validate one InitialParameterization assignment against declarations."""
    if variable_name not in var_map:
        return [
            GNNParseError(
                code="GNN-W003",
                message=f"Parameterization for undeclared variable '{variable_name}'",
                line=end_line,
                file=file_path,
                severity="warning",
            )
        ]

    decl = var_map[variable_name]
    expected_shape = _numeric_dimensions(decl.dimensions)
    if expected_shape is None:
        logger.debug(
            "Symbolic dimensions for '%s', skipping numeric check: %s",
            variable_name,
            decl.dimensions,
        )
        return []

    try:
        parsed_value = _parse_parameter_value(raw_value)
    except (SyntaxError, ValueError, TypeError) as e:
        logger.debug(
            "Could not parse parameterization for '%s', skipping shape check: %s",
            variable_name,
            e,
        )
        return []

    actual_shape, ragged = _parameter_shape(parsed_value)
    comparable_shape = _shape_for_comparison(actual_shape, expected_shape)
    if comparable_shape == expected_shape and not ragged:
        return []

    ragged_suffix = " (ragged)" if ragged else ""
    return [
        GNNParseError(
            code="GNN-E002",
            message=(
                f"Matrix '{variable_name}': declared shape "
                f"{_format_shape(expected_shape)} but parameterization has shape "
                f"{_format_shape(comparable_shape)}{ragged_suffix}"
            ),
            line=end_line,
            file=file_path,
        )
    ]


def validate_matrix_dimensions(
    content: str,
    variables: List[GNNVariable],
    *,
    file_path: Optional[str] = None,
) -> List[GNNParseError]:
    """
    Cross-validate InitialParameterization dimensions against StateSpaceBlock
    declarations.

    Returns list of GNNParseError for any mismatches.
    """
    errors: List[GNNParseError] = []
    var_map = {v.name: v for v in variables}

    in_param = False
    current_var: Optional[str] = None
    brace_depth = 0
    value_lines: list[str] = []

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        if line.startswith("## "):
            if current_var is not None:
                errors.extend(
                    _validate_parameter_assignment(
                        variable_name=current_var,
                        raw_value="\n".join(value_lines),
                        end_line=line_no - 1,
                        var_map=var_map,
                        file_path=file_path,
                    )
                )
                current_var = None
                value_lines = []
                brace_depth = 0
            section_name = line[3:].strip()
            in_param = section_name == "InitialParameterization"
            continue

        if not in_param:
            continue
        if not line or line.startswith("#"):
            continue

        value_line = _strip_parameter_comment(line)
        if not value_line:
            continue

        if current_var is not None:
            value_lines.append(value_line)
            brace_depth += value_line.count("{") - value_line.count("}")
        else:
            assign_match = _PARAM_ASSIGN_RE.match(value_line)
            if not assign_match:
                continue
            current_var = assign_match.group("name")
            assignment_value = assign_match.group("value").strip()
            value_lines = [assignment_value]
            brace_depth = assignment_value.count("{") - assignment_value.count("}")

        if current_var is not None and brace_depth <= 0:
            errors.extend(
                _validate_parameter_assignment(
                    variable_name=current_var,
                    raw_value="\n".join(value_lines),
                    end_line=line_no,
                    var_map=var_map,
                    file_path=file_path,
                )
            )
            current_var = None
            value_lines = []
            brace_depth = 0

    if current_var is not None:
        errors.extend(
            _validate_parameter_assignment(
                variable_name=current_var,
                raw_value="\n".join(value_lines),
                end_line=len(content.splitlines()),
                var_map=var_map,
                file_path=file_path,
            )
        )

    return errors


# ─── JSON Schema ─────────────────────────────────────────────────────────────────

GNN_MODEL_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://activeinference.org/gnn/model.schema.json",
    "title": "GNN Model",
    "description": "Schema for a parsed GNN (Generalized Notation Notation) model object.",
    "type": "object",
    "required": [
        "gnn_section",
        "gnn_version",
        "model_name",
        "state_space",
        "connections",
    ],
    "properties": {
        "gnn_section": {
            "type": "string",
            "description": "Short model identifier (no spaces).",
            "pattern": r"^[A-Za-z][A-Za-z0-9_]*$",
        },
        "gnn_version": {
            "type": "string",
            "description": "GNN language version.",
            "enum": ["1", "1.0", "1.1"],
        },
        "model_name": {
            "type": "string",
            "description": "Human-readable model name.",
            "minLength": 1,
        },
        "model_annotation": {
            "type": "string",
            "description": "Optional free-text model description.",
        },
        "state_space": {
            "type": "array",
            "description": "List of variable declarations.",
            "items": {
                "type": "object",
                "required": ["name", "dimensions", "dtype"],
                "properties": {
                    "name": {"type": "string"},
                    "dimensions": {
                        "type": "array",
                        "items": {"type": ["string", "integer"]},
                    },
                    "dtype": {"type": "string", "enum": ["float", "int", "bool"]},
                    "default": {"type": ["string", "null"]},
                },
            },
        },
        "connections": {
            "type": "array",
            "description": "List of edges between state-space variables.",
            "items": {
                "type": "object",
                "required": ["source", "target", "directed"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "directed": {"type": "boolean"},
                    "label": {"type": ["string", "null"]},
                },
            },
        },
        "initial_parameterization": {
            "type": "object",
            "description": "Variable name → matrix values (as nested lists).",
            "additionalProperties": True,
        },
    },
}


def validate_gnn_object(obj: Dict[str, Any]) -> List[str]:
    """
    Validate a parsed GNN model dict against the JSON Schema.

    Uses jsonschema if available, otherwise performs basic key checks.

    Returns:
        List of validation error messages (empty = valid).
    """
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(GNN_MODEL_SCHEMA)
        return [e.message for e in sorted(validator.iter_errors(obj), key=str)]
    except ImportError:
        logger.debug("jsonschema not installed — using basic validation")
        missing = [k for k in GNN_MODEL_SCHEMA["required"] if k not in obj]
        return [f"Missing required key: '{k}'" for k in missing]


# ─── Required Sections Validation ────────────────────────────────────────────────

REQUIRED_SECTIONS: set[Any] = {
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
}


def validate_required_sections(
    content: str,
    *,
    file_path: Optional[str] = None,
) -> List[GNNParseError]:
    """Check that all required sections are present."""
    found: set[Any] = set()
    for line in content.splitlines():
        if line.strip().startswith("## "):
            found.add(line.strip()[3:].strip())

    errors: list[Any] = []
    for section in REQUIRED_SECTIONS:
        if section not in found:
            errors.append(
                GNNParseError(
                    code="GNN-E001",
                    message=f"Missing required section: '## {section}'",
                    file=file_path,
                )
            )
    return errors
