"""
Type checking rules and type registry for GNN models.

Provides the canonical set of valid GNN types, naming patterns,
and validation rule definitions used by the core type checker.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .dimensions import parse_state_variables

# Canonical valid GNN types (Active Inference domain)
VALID_TYPES: list[str] = [
    "int",
    "float",
    "double",
    "string",
    "bool",
    "array",
    "matrix",
    "vector",
    "tensor",
    "state",
    "action",
    "observation",
    "belief",
    "Categorical",
    "Dirichlet",
    "Gaussian",
    "Continuous",
    "Discrete",
    "POMDP",
    "MDP",
    "GenerativeModel",
    "Distribution",
]

# Regex patterns for type validation
TYPE_PATTERNS: Dict[str, str] = {
    "numeric": r"^[0-9]+(\.[0-9]+)?$",
    "identifier": r"^[^\W\d][\w']*(?:\+\d+)?$",
    "array": r"^\[.*\]$",
}


def get_validation_rules() -> Dict[str, Any]:
    """Return the full validation rule set for GNN types.

    Returns:
        Dictionary containing ``valid_types`` and ``type_patterns``.
    """
    return {
        "valid_types": VALID_TYPES,
        "type_patterns": TYPE_PATTERNS,
    }


def validate_type(type_info: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single type definition against the rule set.

    Args:
        type_info: Dict with ``name`` and ``type`` keys.

    Returns:
        Dict with ``valid`` (bool), ``message`` (str), ``variable``, ``type``.
    """
    var_name = type_info["name"]
    var_type = type_info["type"]

    validation: Dict[str, Any] = {
        "valid": True,
        "message": "",
        "variable": var_name,
        "type": var_type,
    }

    if var_type not in VALID_TYPES:
        validation["valid"] = False
        validation["message"] = f"Unknown type '{var_type}' for variable '{var_name}'"

    if not re.match(TYPE_PATTERNS["identifier"], var_name):
        validation["valid"] = False
        validation["message"] = f"Invalid variable name '{var_name}'"

    return validation


def check_type_consistency(types: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Check consistency of types across a file (e.g. duplicate names).

    Args:
        types: List of dicts with ``name`` keys.

    Returns:
        Dict with ``consistent`` (bool) and ``message`` (str).
    """
    consistency: Dict[str, Any] = {
        "consistent": True,
        "message": "",
    }

    seen: set[str] = set()
    duplicates: list[str] = []
    for type_info in types:
        name = str(type_info["name"])
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)

    if duplicates:
        consistency["consistent"] = False
        consistency["message"] = f"Duplicate variable names: {', '.join(duplicates)}"

    return consistency


def extract_types_from_content(content: str) -> list[Dict[str, Any]]:
    """Extract type annotations from raw GNN file content.

    Parses only canonical declarations in ``StateSpaceBlock`` and returns a
    list of ``{name, type, line}`` dictionaries. Prose, parameter values, and
    comments outside that section are deliberately excluded.

    Args:
        content: Full GNN file content as string.

    Returns:
        List of extracted type entries.
    """
    variables, _ = parse_state_variables(content)
    return [
        {
            "name": variable.name,
            "type": variable.dtype,
            "line": variable.line,
        }
        for variable in variables
    ]
