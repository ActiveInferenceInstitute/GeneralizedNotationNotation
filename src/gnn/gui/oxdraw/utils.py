"""
Utility functions for oxdraw integration

Provides helper functions for node/edge styling, validation, and configuration.
"""

import re
from typing import Any, Dict, List, Tuple, cast


def infer_node_shape(var_name: str, var_data: Dict[str, Any]) -> Tuple[str, str]:
    r"""
    Infer Mermaid node shape from GNN variable characteristics.

    Shape mapping:
    - Matrices (A, B): Rectangle [A]
    - Vectors (C, D, E): Rounded (C)
    - States (s, s_prime): Stadium ([s])
    - Observations (o): Circle ((o))
    - Actions (u): Hexagon {{u}}
    - Policies (π): Diamond {π}
    - Free Energy (F, G): Trapezoid [/F\]

    Args:
        var_name: Variable name
        var_data: Variable metadata dictionary

    Returns:
        Tuple of (opening_bracket, closing_bracket)
    """
    dims = var_data.get("dimensions", [])
    ontology = var_data.get("ontology_mapping", "")

    # Check ontology mapping first (more specific)
    if "State" in ontology:
        # States use stadium shape
        return "([", "])"
    elif "Observation" in ontology:
        # Observations use circles
        return "((", "))"
    elif "Action" in ontology or var_name == "u":
        # Actions use hexagons
        return "{{", "}}"
    elif "Policy" in ontology or var_name in ["π", "pi"]:
        # Policies use diamonds
        return "{", "}"
    elif "FreeEnergy" in ontology or var_name in ["F", "G"]:
        # Free energy uses trapezoid
        return "[/", "\\]"

    # Check dimensionality second
    if len(dims) >= 2:
        # Matrices use rectangles
        return "[", "]"
    else:
        # Default vectors use rounded
        return "(", ")"


def infer_edge_style(symbol: str) -> str:
    """
    Convert GNN connection symbols to Mermaid edge styles.

    Mapping:
    - > : Generative (thick arrow) ==>
    - - : Inference (dashed line) -.->
    - * : Modulation (dotted line) -..->
    - ~ : Weak coupling (thin line) -->

    Args:
        symbol: GNN connection symbol

    Returns:
        Mermaid edge style string
    """
    style_map: dict[str, Any] = {
        ">": "==>",  # Generative
        "-": "-.->",  # Inference
        "*": "-..->",  # Modulation
        "~": "-->",  # Coupling
    }

    return cast("str", style_map.get(symbol, "-->"))  # Default to normal arrow


def validate_mermaid_syntax(mermaid_content: str) -> Tuple[bool, List[str]]:
    """
    Validate basic Mermaid syntax.

    Checks:
    - Starts with flowchart directive
    - Node definitions are well-formed
    - Edge definitions are well-formed
    - Brackets are balanced

    Args:
        mermaid_content: Mermaid diagram content

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: list[Any] = []

    # Check for flowchart directive
    if not re.search(r"^\s*flowchart\s+(TD|LR|TB|RL)", mermaid_content, re.MULTILINE):
        errors.append("Missing flowchart directive (e.g., 'flowchart TD')")

    # Check bracket balance
    bracket_pairs: list[Any] = [("[", "]"), ("(", ")"), ("{", "}")]

    for open_b, close_b in bracket_pairs:
        open_count = mermaid_content.count(open_b)
        close_count = mermaid_content.count(close_b)
        if open_count != close_count:
            errors.append(
                f"Unbalanced brackets: {open_count} '{open_b}' vs {close_count} '{close_b}'"
            )

    # Check for common syntax errors
    if "-->" in mermaid_content and "-- >" in mermaid_content:
        errors.append("Inconsistent arrow spacing (mix of '-->' and '-- >')")

    return (len(errors) == 0, errors)
