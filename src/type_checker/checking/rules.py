"""
Type checking rules and type registry for GNN models.

Provides the canonical set of valid GNN types, naming patterns,
and validation rule definitions used by the core type checker.
"""

from __future__ import annotations

import re
from typing import Any, Dict


# Canonical valid GNN types (Active Inference domain)
VALID_TYPES: list[str] = [
    "int", "float", "double", "string", "bool", "array", "matrix",
    "vector", "tensor", "state", "action", "observation", "belief",
    "Categorical", "Dirichlet", "Gaussian", "Continuous", "Discrete",
    "POMDP", "MDP", "GenerativeModel", "Distribution",
]

# Regex patterns for type validation
TYPE_PATTERNS: Dict[str, str] = {
    "numeric": r"^[0-9]+(\.[0-9]+)?$",
    "identifier": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    "array": r"^\[.*\]$",
}

# Regex patterns for extracting type annotations from GNN content
EXTRACTION_PATTERNS: list[str] = [
    r'([^#\w])(\w+)\s*:\s*([a-zA-Z0-9_]+)',          # name: type (excluding comments)
    r'(\w+)\s*\[(?:[^\]]*?)type=([a-zA-Z0-9_]+)(?:[^\]]*?)\]',  # name[...type=float...]
    r'(\w+)\s*\[([0-9\s,]+)\]',                        # name[dimensions] (pure numbers as shapes)
]


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

    var_names = [t["name"] for t in types]
    duplicates = [name for name in set(var_names) if var_names.count(name) > 1]

    if duplicates:
        consistency["consistent"] = False
        consistency["message"] = f"Duplicate variable names: {', '.join(duplicates)}"

    return consistency


def extract_types_from_content(content: str) -> list[Dict[str, Any]]:
    """Extract type annotations from raw GNN file content.

    Applies all ``EXTRACTION_PATTERNS`` against *content* and returns
    a list of ``{name, type, line}`` dicts.

    Args:
        content: Full GNN file content as string.

    Returns:
        List of extracted type entries.
    """
    found_types: list[Dict[str, Any]] = []
    for pattern in EXTRACTION_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            if len(match.groups()) >= 3 and pattern.startswith(r'([^#'):
                var_name = match.group(2)
                var_type = match.group(3)
            elif len(match.groups()) >= 2:
                var_name = match.group(1)
                var_type = match.group(2)
            else:
                continue

            found_types.append({
                "name": var_name,
                "type": var_type,
                "line": content[:match.start()].count('\n') + 1,
            })

    return found_types
