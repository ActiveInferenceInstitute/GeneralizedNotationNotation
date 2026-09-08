#!/usr/bin/env python3
"""
Categorical DisCoPy diagram builders for GNN Step 11.

Extracted from ``render.discopy.translator``.
"""

import functools
import logging
import re
from typing import Optional

from .bootstrap import (
    TENSOR_COMPONENTS_AVAILABLE,
    Box,
    Diagram,
    Dim,
    Id,
)
from .gnn_parsing import _parse_dims_str

logger = logging.getLogger(__name__)


def gnn_statespace_to_discopy_dims_map(parsed_gnn: dict) -> dict[str, Dim]:
    """
    Converts GNN StateSpaceBlock entries into a dictionary mapping variable names to DisCoPy Dim objects.
    Handles parsing of dimensions like VarName[dim1,dim2,...] or VarName.
    """
    if not TENSOR_COMPONENTS_AVAILABLE or Dim is None:
        logger.error(
            "DisCoPy Dim component is not available. Cannot create Dim objects."
        )
        logger.info("Run generate_setup_report() for installation instructions")
        return {}

    dims_map: dict[str, Dim] = {}
    statespace_lines = parsed_gnn.get("StateSpaceBlock", [])
    if not statespace_lines:
        logger.warning(
            "StateSpaceBlock not found or empty. No DisCoPy Dim objects will be created."
        )
        return dims_map

    # Pattern for variable names and optional dimensions: VarName[dim1,dim2,...] or VarName[dim]
    # Dimensions are comma-separated integers. Ignores content like type=...
    var_pattern = re.compile(
        r"^\s*([a-zA-Z_π][\w_π]*)\s*(?:\x5B([^\x5D]*)\x5D)?\s*(?:#.*)?$"
    )

    for line in statespace_lines:
        line_content = line.strip()
        if not line_content or line_content.startswith("#"):
            continue

        match = var_pattern.match(line_content)
        if not match:
            logger.warning(
                f"Could not parse variable from StateSpaceBlock line: '{line_content}'. Skipping."
            )
            continue

        var_name = match.group(1)
        dims_str = match.group(2)

        parsed_dims_list = _parse_dims_str(
            dims_str
        )  # _parse_dims_str returns list[int], e.g., [1] or [2,3]

        try:
            # Create Dim object using the parsed integer dimensions
            # Dim(*[1]) creates Dim(1), Dim(*[2,3]) creates Dim(2,3)
            current_dim = Dim(*parsed_dims_list)
            dims_map[var_name] = current_dim
            logger.debug(
                f"Created DisCoPy Dim: {current_dim} for GNN variable '{var_name}'"
            )
        except (
            Exception
        ) as e_dim_creation:  # Catch potential errors during Dim creation
            logger.error(
                f"Error creating DisCoPy Dim for '{var_name}' with dims {parsed_dims_list}: {e_dim_creation}"
            )
            continue

    return dims_map


def gnn_connections_to_discopy_diagram(
    parsed_gnn: dict, dims_map: dict[str, Dim]
) -> Optional[Diagram]:
    """
    Converts GNN Connections into a DisCoPy Diagram (from discopy.tensor).
    This version creates a diagram with abstract boxes (no tensor data).
    """
    # Check if essential DisCoPy components are available
    if not TENSOR_COMPONENTS_AVAILABLE or any(
        comp is None for comp in [Diagram, Box, Id, Dim]
    ):
        logger.error(
            "Core DisCoPy components (Diagram, Box, Id, Dim) are not available. Cannot create tensor-based DisCoPy diagram."
        )
        logger.info("Run generate_setup_report() for installation instructions")
        return None

    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning(
            "Connections section not found or empty. Cannot create DisCoPy diagram."
        )
        return None

    diagram: Diagram = (
        Id()
    )  # Start with an Id for tensor diagrams, explicitly type hint diagram

    # Regex patterns (copied from original, ensure they are correct for this context)
    var_id_pattern = r"[a-zA-Z_π][\w_π]*"

    # Pattern for a list of one or more comma-separated variable names:
    # e.g., "Var1", "Var1, Var2", "Var1, Var2, Var3"
    # This pattern itself does not match surrounding parentheses.
    var_list_content_pattern = var_id_pattern + r"(?:\s*,\s*" + var_id_pattern + r")*"

    # Pattern for a block of text that forms one side of a connection (source or target).
    # This matches EITHER a parenthesized list OR a non-parenthesized list.
    # Example: matches "( Var1, Var2 )" OR "Var1, Var2".
    (
        # Option 1: ( list of vars ) - captures list content in a group
        # Using \( and \) for literal parentheses.
        r"(?:\s*\(\s*("
        + var_list_content_pattern
        + r")\s*\)\s*|"
        +
        # Option 2: list of vars
        r"\s*("
        + var_list_content_pattern
        + r")\s*)"
    )

    # Final connection pattern string.
    # It captures the source block (group 1 for parenthesized, group 2 for non-parenthesized)
    # and target block (group 3 for parenthesized, group 4 for non-parenthesized)
    # Supports '>', '->', '-' as connectors.
    # Ignores '=' for assignments for now.
    conn_pattern_str = (
        # Source part: matches either a parenthesized list or a direct list/single var
        # Using \( and \) for literal parentheses.
        r"^\s*(?:\(\s*("
        + var_list_content_pattern
        + r")\s*\)|("
        + var_list_content_pattern
        + r"))\s*"
        +
        # Connector
        r"(?:>|->|-)\s*"
        +
        # Target part: matches either a parenthesized list or a direct list/single var
        r"(?:\(\s*("
        + var_list_content_pattern
        + r")\s*\)|("
        + var_list_content_pattern
        + r"))\s*(?:#.*)?$"
    )
    conn_pattern = re.compile(conn_pattern_str)

    # Pattern for assignment-like connections (e.g., G=ExpectedFreeEnergy, t=Time)
    # These are treated as non-diagrammatic for now. Ensuring this uses \w and \s.
    assignment_pattern_str = r"^\s*([a-zA-Z_π][\w_π]*)\s*=\s*([^#]+?)\s*(?:#.*)?$"
    assignment_pattern = re.compile(assignment_pattern_str)

    def parse_vars_from_group(group_str: str | None) -> list[str]:
        """Parse vars from group."""
        if not group_str:
            return []
        # Handles cases where the string might already be clean or needs splitting.
        return [v.strip() for v in group_str.split(",") if v.strip()]

    for line in connections_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        assignment_match = assignment_pattern.match(line)
        if assignment_match:
            var_name = assignment_match.group(1)
            value_assigned = assignment_match.group(2).strip()
            logger.info(
                f"Parsed assignment: '{var_name}' = '{value_assigned}'. Not creating a DisCoPy box for this."
            )
            # Optionally, these could be stored as annotations on the diagram or nodes if relevant
            continue  # Move to next line

        match = conn_pattern.match(line)
        if match:
            # Determine actual source and target strings from the four possible capture groups
            # Groups are: 1 (source parenthesized), 2 (source direct), 3 (target parenthesized), 4 (target direct)
            source_str_paren = match.group(1)
            source_str_direct = match.group(2)
            target_str_paren = match.group(3)
            target_str_direct = match.group(4)

            source_content = source_str_paren if source_str_paren else source_str_direct
            target_content = target_str_paren if target_str_paren else target_str_direct

            if not source_content or not target_content:
                logger.warning(
                    f"Could not determine source or target content from connection line: '{line}'. Skipping."
                )
                continue

            source_vars = parse_vars_from_group(source_content)
            target_vars = parse_vars_from_group(target_content)

            if not source_vars or not target_vars:
                logger.warning(
                    f"Empty source or target variables after parsing connection: '{line}'. Skipping."
                )
                continue

            # Validate all variables exist in types
            all_vars_valid = True
            for var_list in [source_vars, target_vars]:
                for var_name in var_list:
                    if var_name not in dims_map:
                        logger.warning(
                            f"Unknown variable '{var_name}' (not in dims_map) in connection: '{line}'. Skipping connection."
                        )
                        all_vars_valid = False
                        break
                if not all_vars_valid:
                    break
            if not all_vars_valid:
                continue

            # Determine domain and codomain types (which are Dim objects)
            if len(source_vars) == 1:
                dom_dim = dims_map[source_vars[0]]
            elif len(source_vars) > 1:
                # Tensor product of Dim objects
                try:
                    dom_dim = functools.reduce(
                        lambda a, b: a @ b, [dims_map[v] for v in source_vars]
                    )
                except TypeError:
                    logger.error(
                        f"Cannot compute tensor product for domain with vars {source_vars}. Skipping connection."
                    )
                    continue
            else:  # No source vars, use empty Dim (identity for tensor product, usually Dim(1))
                dom_dim = Dim()

            if len(target_vars) == 1:
                cod_dim = dims_map[target_vars[0]]
            elif len(target_vars) > 1:
                try:
                    cod_dim = functools.reduce(
                        lambda a, b: a @ b, [dims_map[v] for v in target_vars]
                    )
                except TypeError:
                    logger.error(
                        f"Cannot compute tensor product for codomain with vars {target_vars}. Skipping connection."
                    )
                    continue
            else:  # No target vars, use empty Dim
                cod_dim = Dim()

            source_name_part = "_".join(source_vars)
            target_name_part = "_".join(target_vars)
            box_name = f"{source_name_part}_to_{target_name_part}"

            box = Box(
                box_name, dom_dim, cod_dim
            )  # Box expects Dim for dom/cod in discopy.tensor
            logger.debug(
                f"Created DisCoPy Box: Box('{box_name}', dom={dom_dim}, cod={cod_dim})"
            )

            # Simple sequential composition for now
            if (
                diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes
            ):  # First box, Dim() is the domain/codomain of Id()
                diagram = box
            elif diagram.cod == dom_dim:  # Chainable
                diagram = diagram >> box
            else:
                # This indicates a more complex structure (e.g. parallel wires or new starting chain)
                # For now, we will log a warning and try to append it as a new parallel component
                logger.warning(
                    f"Connection from '{source_content}' to '{target_content}' (Box dom={dom_dim}, cod={cod_dim}) does not directly chain with previous diagram codomain ({diagram.cod}). Appending in parallel (basic)."
                )
                # Attempting a parallel composition; this assumes variables are distinct flows if not chained.
                # A more robust solution would analyze the full graph structure.
                try:
                    diagram = diagram @ box
                except Exception as e_parallel:
                    logger.error(
                        f"Failed to compose Box('{box_name}') in parallel: {e_parallel}. Diagram construction may be incorrect."
                    )
                    return diagram  # Return what we have so far
        else:
            logger.warning(
                f"Could not parse Connections line: '{line}'. Supported format: 'Source > Target'."
            )

    if (
        diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes
    ):  # Check if any boxes were actually added
        logger.warning("No valid connections were parsed to form a Diagram.")
        return None

    return diagram
