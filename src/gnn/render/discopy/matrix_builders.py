#!/usr/bin/env python3
"""
Matrix-backed DisCoPy diagram builders for GNN Step 11.

Extracted from ``render.discopy.translator``.
"""

import logging
import re
from typing import (
    Any,
    Callable,
    Optional,
)

from .bootstrap import (
    JAX_AVAILABLE,
    TENSOR_COMPONENTS_AVAILABLE,
    Box,
    Diagram,
    Dim,
    Id,
    Matrix,
    discopy_backend,
    jax,
    jnp,
)
from .gnn_parsing import _convert_json_to_complex_array

logger = logging.getLogger(__name__)


def gnn_connections_to_discopy_matrix_diagram(
    parsed_gnn: dict,
    dims_map: dict[str, Dim],
    tensor_definitions: dict,
    prng_key_provider: Optional[Callable[[str], Any]] = None,
    default_dtype_str: str = "float32",
) -> Optional[Diagram]:
    """
    Converts GNN Connections into a DisCoPy Diagram, where boxes are populated with Matrix objects
    (from discopy.matrix) containing JAX-backed Tensors if JAX is available.
    If JAX is not available or tensor data is missing, boxes may be abstract.
    """
    if not JAX_AVAILABLE:  # Check the overall JAX_AVAILABLE flag
        logger.error(
            "JAX or essential DisCoPy components for matrix operations are not available. Cannot create a JAX-backed MatrixDiagram."
        )
        logger.info("Run generate_setup_report() for installation instructions")
        return None

    # Also ensure all required components are available
    if (
        not TENSOR_COMPONENTS_AVAILABLE
        or any(comp is None for comp in [Diagram, Box, Id, Dim, Matrix])
        or discopy_backend is None
        or jax is None
        or jnp is None
    ):
        logger.error(
            "Critical JAX/DisCoPy components (Diagram, Box, Id, Dim, Matrix, jax, jnp, backend) are not available. Cannot create MatrixDiagram."
        )
        logger.info("Run generate_setup_report() for installation instructions")
        return None

    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning(
            "Connections section not found or empty. Cannot create DisCoPy MatrixDiagram."
        )
        return None

    diagram: Diagram = Id()  # Start with an Id, explicitly type hint

    # Regex patterns (ensure they are correct for this context)
    var_id_pattern = r"[a-zA-Z_π][\w_π]*"

    # Pattern for a list of one or more comma-separated variable names:
    # e.g., "Var1", "Var1, Var2", "Var1, Var2, Var3"
    # This pattern itself does not match surrounding parentheses.
    var_list_content_pattern = var_id_pattern + r"(?:\\s*,\\s*" + var_id_pattern + r")*"

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
    assignment_pattern_str = r"^\\s*([a-zA-Z_π][\\w_π]*)\\s*=\\s*([^#]+?)\\s*(?:#.*)?$"
    assignment_pattern = re.compile(assignment_pattern_str)

    def parse_vars_from_group(group_str: str | None) -> list[str]:
        """Parse vars from group."""
        if not group_str:
            return []
        return [v.strip() for v in group_str.split(",") if v.strip()]

    # Initialize with an empty diagram or appropriate identity
    diagram = Id(Dim(1))  # Start with Identity on Dim(1) for matrix diagrams

    tensor_definitions = parsed_gnn.get("TensorDefinitions", {})
    # Get the raw dtype definition, which might be a string or a type object
    raw_default_dtype = tensor_definitions.get("default_dtype", "float32")

    # Ensure default_dtype_str is a string name
    if isinstance(raw_default_dtype, str):
        default_dtype_str = raw_default_dtype
    elif hasattr(
        raw_default_dtype, "__name__"
    ):  # Check if it's a type object with a name
        default_dtype_str = raw_default_dtype.__name__
    else:
        logger.warning(
            f"Unrecognized default_dtype format: {raw_default_dtype}. Defaulting to 'float32'."
        )
        default_dtype_str = "float32"

    logger.debug(f"Processed default_dtype_str: {default_dtype_str}")

    # Attempt to get the actual JAX dtype object using the string name
    if JAX_AVAILABLE and jnp and hasattr(jnp, default_dtype_str):
        jax_dtype = getattr(jnp, default_dtype_str)
    else:
        # Recovery if JAX not available or dtype string is not a jnp attribute.
        jax_dtype = default_dtype_str
        logger.debug(
            f"JAX/jnp not fully available or '{default_dtype_str}' not in jnp. Using dtype name '{jax_dtype}'."
        )

    for line_idx, line in enumerate(connections_lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        assignment_match = assignment_pattern.match(line)
        if assignment_match:
            # Assignments are not typically part of MatrixDiagram structure, logged and skipped.
            logger.info(
                f"Skipping assignment in Connections for MatrixDiagram: '{line}'"
            )
            continue

        match = conn_pattern.match(line)
        if match:
            source_str_paren, source_str_direct, target_str_paren, target_str_direct = (
                match.groups()
            )
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
                    f"Empty source or target variables for MatrixDiagram: '{line}'. Skipping."
                )
                continue

            all_vars_valid = True
            for var_list in [source_vars, target_vars]:
                for var_name in var_list:
                    if var_name not in dims_map:
                        logger.warning(
                            f"Unknown variable '{var_name}' (no Dim found) in connection: '{line}'. Skipping."
                        )
                        all_vars_valid = False
                        break
                if not all_vars_valid:
                    break
            if not all_vars_valid:
                continue

            dom_dim = dims_map[source_vars[0]] if len(source_vars) == 1 else Dim()
            if len(source_vars) > 1:
                current_dom_dim = dims_map[source_vars[0]]
                for i in range(1, len(source_vars)):
                    current_dom_dim = current_dom_dim @ dims_map[source_vars[i]]
                dom_dim = current_dom_dim
            elif not source_vars:
                continue

            cod_dim = dims_map[target_vars[0]] if len(target_vars) == 1 else Dim()
            if len(target_vars) > 1:
                current_cod_dim = dims_map[target_vars[0]]
                for i in range(1, len(target_vars)):
                    current_cod_dim = current_cod_dim @ dims_map[target_vars[i]]
                cod_dim = current_cod_dim
            elif not target_vars:
                continue

            box_name_short = f"{'_'.join(source_vars)}_to_{'_'.join(target_vars)}"
            box_name_full_line = (
                line  # Or some unique identifier for the box from this line
            )

            # Retrieve tensor data
            tensor_def = tensor_definitions.get(box_name_short)  # Try short name first
            if not tensor_def:
                tensor_def = tensor_definitions.get(
                    box_name_full_line
                )  # Try full line if specific

            if not tensor_def:
                logger.warning(
                    f"No tensor definition found for box '{box_name_short}' or full line '{box_name_full_line}'. Skipping box."
                )
                continue

            # Determine data type for JAX array for this specific box
            raw_box_dtype = tensor_def.get(
                "dtype", default_dtype_str
            )  # Inherit default if not specified

            if isinstance(raw_box_dtype, str):
                box_dtype_str = raw_box_dtype
            elif hasattr(raw_box_dtype, "__name__"):
                box_dtype_str = raw_box_dtype.__name__
            else:
                logger.warning(
                    f"Unrecognized dtype format for box '{box_name_short}': {raw_box_dtype}. Using default: '{default_dtype_str}'."
                )
                box_dtype_str = default_dtype_str

            logger.debug(
                f"Processed box_dtype_str for '{box_name_short}': {box_dtype_str}"
            )

            if JAX_AVAILABLE and jnp and hasattr(jnp, box_dtype_str):
                current_jax_dtype = getattr(jnp, box_dtype_str)
            else:
                current_jax_dtype = box_dtype_str
                logger.debug(
                    f"JAX/jnp not fully available or '{box_dtype_str}' not in jnp for box '{box_name_short}'. Using dtype name '{current_jax_dtype}'."
                )

            initializer = tensor_def.get("initializer")
            jax_array_data = None

            # Ensure dom_dim.inside and cod_dim.inside are tuples for concatenation
            dom_inside_tuple = (
                tuple(dom_dim.inside) if hasattr(dom_dim, "inside") else dom_dim.inside
            )
            cod_inside_tuple = (
                tuple(cod_dim.inside) if hasattr(cod_dim, "inside") else cod_dim.inside
            )

            if not isinstance(dom_inside_tuple, tuple) or not isinstance(
                cod_inside_tuple, tuple
            ):
                logger.error(
                    f"Cannot determine box shape for '{box_name_short}'. Expected .inside to be tuples, got {type(dom_inside_tuple)} and {type(cod_inside_tuple)}. Skipping."
                )
                continue

            box_shape = dom_inside_tuple + cod_inside_tuple

            if isinstance(initializer, list):  # Direct data from JSON
                logger.debug(
                    f"MatrixDiagram: Initializer for '{box_name_short}' IS a list. Processing with _convert_json_to_complex_array."
                )
                try:
                    # Convert [real, imag] pairs to complex numbers before creating JAX array
                    processed_initializer = _convert_json_to_complex_array(initializer)

                    if JAX_AVAILABLE and jnp:
                        # Ensure jax_dtype is a JAX dtype object if complex data is detected
                        # This is a bit of a heuristic; ideally, dtype comes from GNN or is more robustly inferred.
                        if any(
                            isinstance(x, complex) for x in processed_initializer
                        ) or (
                            isinstance(processed_initializer, list)
                            and processed_initializer
                            and any(
                                isinstance(x, complex)
                                for row in processed_initializer
                                if isinstance(row, list)
                                for x in row
                            )
                        ):  # check nested for complex
                            if (
                                isinstance(current_jax_dtype, str)
                                and "complex" not in current_jax_dtype.lower()
                            ):
                                logger.debug(
                                    f"Initializer for '{box_name_short}' contains complex numbers. Overriding dtype to jnp.complex64 from {current_jax_dtype}."
                                )
                                current_jax_dtype = (
                                    jnp.complex64
                                    if hasattr(jnp, "complex64")
                                    else "complex64"
                                )
                            elif not hasattr(
                                current_jax_dtype, "is_complex"
                            ):  # if it's already a jax dtype, check if complex
                                if not jnp.issubdtype(
                                    current_jax_dtype, jnp.complexfloating
                                ):
                                    logger.debug(
                                        f"Initializer for '{box_name_short}' contains complex numbers. Overriding dtype to jnp.complex64 from {current_jax_dtype}."
                                    )
                                    current_jax_dtype = (
                                        jnp.complex64
                                        if hasattr(jnp, "complex64")
                                        else "complex64"
                                    )

                        jax_array_data = jnp.array(
                            processed_initializer, dtype=current_jax_dtype
                        ).reshape(box_shape)
                    else:
                        logger.warning(
                            "JAX not available, attempting to create NumPy array for MatrixBox data from list. This path might not be fully supported for MatrixDiagrams."
                        )
                        # MatrixBox itself might still expect a JAX tensor if DisCoPy is in JAX mode,
                        # this is more of a graceful degradation attempt.
                        # The proper fix is to not call this function if JAX is not available.
                        # For now, we make a numpy array, but this won't work if a JAX tensor is strictly required by DisCoPy.
                        import numpy  # Local import for this recovery

                        jax_array_data = numpy.array(
                            processed_initializer, dtype=default_dtype_str
                        ).reshape(box_shape)

                except Exception as e:
                    logger.error(
                        f"Failed to create JAX array from literal for '{box_name_short}': {e}. Shape: {box_shape}, Init: {processed_initializer}"
                    )
                    continue
            elif isinstance(initializer, str):
                logger.debug(
                    f"MatrixDiagram: Initializer for '{box_name_short}' IS a string: '{initializer}'. Checking for load/random."
                )
                if initializer.startswith("load:"):
                    file_path_str = initializer[len("load:") :]
                    try:
                        # Ensure path is absolute or resolve relative to GNN file (if context available)
                        # For now, assume path is resolvable as is or relative to where script runs
                        loaded_np_array = numpy.load(file_path_str)
                        jax_array_data = jnp.array(
                            loaded_np_array, dtype=jax_dtype
                        ).reshape(box_shape)
                    except Exception as e:
                        logger.error(
                            f"Failed to load JAX array from '{file_path_str}' for '{box_name_short}': {e}"
                        )
                        continue
                elif (
                    initializer.startswith("random_normal:")
                    or initializer == "random_normal"
                ):
                    if JAX_AVAILABLE and jax and prng_key_provider:
                        key_suffix = (
                            initializer.split(":", 1)[1]
                            if ":" in initializer
                            else str(line_idx)
                        )
                        current_key = prng_key_provider(
                            f"{box_name_short}_{key_suffix}"
                        )
                        jax_array_data = jax.random.normal(
                            current_key, shape=box_shape, dtype=jax_dtype
                        )
                    else:
                        logger.warning(
                            f"JAX or PRNG key provider not available for random_normal initializer of '{box_name_short}'. Skipping."
                        )
                        continue
                elif (
                    initializer.startswith("random_uniform:")
                    or initializer == "random_uniform"
                ):
                    if JAX_AVAILABLE and jax and prng_key_provider:
                        key_suffix = (
                            initializer.split(":", 1)[1]
                            if ":" in initializer
                            else str(line_idx)
                        )
                        current_key = prng_key_provider(
                            f"{box_name_short}_{key_suffix}"
                        )
                        jax_array_data = jax.random.uniform(
                            current_key, shape=box_shape, dtype=jax_dtype
                        )
                    else:
                        logger.warning(
                            f"JAX or PRNG key provider not available for random_uniform initializer of '{box_name_short}'. Skipping."
                        )
                        continue
                # Add more random initializers (e.g., glorot, he) as needed
                else:
                    logger.warning(
                        f"Unknown string initializer for '{box_name_short}': '{initializer}'. Skipping."
                    )
                    continue
            else:  # E.g. dict for future complex initializers, or number for scalar broadcast
                if isinstance(initializer, (int, float)):  # Scalar broadcast
                    try:
                        if JAX_AVAILABLE and jnp:
                            jax_array_data = jnp.full(
                                box_shape, initializer, dtype=jax_dtype
                            )
                        else:
                            logger.warning(
                                "JAX not available, attempting to create NumPy full array for MatrixBox data from scalar. This path might not be fully supported."
                            )
                            import numpy  # Local import

                            jax_array_data = numpy.full(
                                box_shape, initializer, dtype=default_dtype_str
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to broadcast scalar for '{box_name_short}': {e}. Shape: {box_shape}, Scalar: {initializer}"
                        )
                        continue
                else:
                    logger.warning(
                        f"Unsupported initializer type for '{box_name_short}': {type(initializer)}. Skipping."
                    )
                    continue

            if jax_array_data is None:
                logger.warning(
                    f"Could not initialize data for box '{box_name_short}'. Skipping."
                )
                continue

            # Create a discopy.matrix.Matrix object to hold the JAX array
            # This Matrix IS a Box, so it can be directly used in the diagram.
            try:
                # Ensure dom_dim.inside and cod_dim.inside are tuples for concatenation.
                box_shape_tuple = tuple(getattr(dom_dim, "inside", ())) + tuple(
                    getattr(cod_dim, "inside", ())
                )

                if not all(isinstance(d, int) for d in box_shape_tuple):
                    logger.error(
                        f"Box '{box_name_short}' has non-integer dimensions in shape: {box_shape_tuple}. Dom: {dom_dim}, Cod: {cod_dim}. Skipping."
                    )
                    continue

                reshaped_jax_array = jax_array_data.reshape(
                    box_shape_tuple
                )  # Reshape based on combined dom and cod dims

                # Create the Matrix (which is a Box)
                logger.debug(f"Preparing to create Matrix for box '{box_name_short}'.")
                logger.debug(f"  dom_dim: {dom_dim} (type: {type(dom_dim)})")
                logger.debug(f"  cod_dim: {cod_dim} (type: {type(cod_dim)})")
                if hasattr(reshaped_jax_array, "shape") and hasattr(
                    reshaped_jax_array, "dtype"
                ):
                    logger.debug(
                        f"  reshaped_jax_array: shape={reshaped_jax_array.shape}, dtype={reshaped_jax_array.dtype} (type: {type(reshaped_jax_array)})"
                    )
                else:
                    logger.debug(
                        f"  reshaped_jax_array: (type: {type(reshaped_jax_array)}), attributes like shape/dtype might be missing."
                    )

                box = Matrix(
                    dom_dim, cod_dim, reshaped_jax_array
                )  # discopy.matrix.Matrix
                # Manually set the name for the Matrix/Box if not automatically handled by constructor in all versions
                box.name = box_name_short

            except Exception as e_matrix_creation:
                logger.error(
                    f"Error creating discopy.matrix.Matrix for box '{box_name_short}': {e_matrix_creation}. Dom: {dom_dim}, Cod: {cod_dim}, Data shape: {jax_array_data.shape if hasattr(jax_array_data, 'shape') else 'N/A'}"
                )
                continue

            # Use box_name_short for logging because box.name may not be set when construction fails early.
            box_data_shape_log = (
                getattr(box.data, "shape", "unknown")
                if hasattr(box, "data") and box.data is not None
                else "no data"
            )
            logger.debug(
                f"Created JAX-backed Matrix: '{box_name_short}', dom={box.dom}, cod={box.cod}, data_shape={box_data_shape_log}"
            )

            if (
                diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes
            ):  # First box, Dim() is the domain/codomain of Id()
                diagram = box
            elif diagram.cod == dom_dim:  # Chainable
                diagram = diagram >> box
            else:
                logger.warning(
                    f"Connection for MatrixBox '{box_name_short}' (dom={dom_dim}) doesn't match diagram cod ({diagram.cod}). Appending in parallel."
                )
                try:
                    diagram = diagram @ box
                except Exception as e_parallel:
                    logger.error(
                        f"Failed to compose MatrixBox '{box_name_short}' in parallel: {e_parallel}. Diagram construction may be incorrect."
                    )
                    return diagram
        else:
            logger.warning(
                f"Could not parse Connections line for MatrixDiagram: '{line}'."
            )

    if diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes:
        logger.warning("No valid connections were parsed to form a MatrixDiagram.")
        return None

    return diagram
