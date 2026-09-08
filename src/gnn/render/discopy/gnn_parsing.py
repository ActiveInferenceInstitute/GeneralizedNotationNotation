#!/usr/bin/env python3
"""
GNN content parsing helpers for GNN Step 11 DisCoPy rendering.

Extracted from ``render.discopy.translator``.
"""

import json
import logging
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)


def _convert_json_to_complex_array(data: Any) -> Any:
    """
    Recursively converts a list structure (from JSON) potentially containing [real, imag] pairs
    into a structure of Python complex numbers, suitable for jnp.array.
    """
    if isinstance(data, list):
        # Check if it's a [real, imag] pair that should be converted to a complex number
        if (
            len(data) == 2
            and isinstance(data[0], (int, float))
            and isinstance(data[1], (int, float))
        ):
            # This condition identifies a list of two numbers as a complex pair.
            return complex(data[0], data[1])
        else:
            # It's a list, but not a [real, imag] pair itself.
            # Recursively process its elements.
            return [_convert_json_to_complex_array(item) for item in data]
    # If data is not a list (e.g., it's a number, string, or already a complex number from a previous step), return it as is.
    return data


def _parse_dims_str(dims_str: str | None) -> list[int]:
    """Helper to parse a comma-separated string of dimensions into a list of ints.
    Extracts leading numeric dimensions, ignoring non-numeric parts like 'type=...'.
    """
    if not dims_str:
        return [1]  # Default to Dim(1) if no dimensions specified

    parsed_dims: list[Any] = []
    # Split by comma, then attempt to convert each part to int.
    # Only add to parsed_dims if it's a valid integer.
    # This will effectively ignore parts like "type=A" or "foo=bar".
    for part in dims_str.split(","):
        stripped_part = part.strip()
        if not stripped_part:  # Skip empty parts
            continue
        try:
            # Try to convert to int. If it works, it's a dimension.
            num = int(stripped_part)
            parsed_dims.append(num)
        except ValueError:
            # This part is not a simple integer (e.g., "type=A"). Log and ignore for dims.
            logger.debug(
                f"Ignoring non-integer part '{stripped_part}' while parsing dimensions from '{dims_str}'"
            )

    if not parsed_dims:  # If no numeric dimensions were found (e.g., just "type=A")
        return [1]  # Default to Dim(1)
    return parsed_dims


def parse_gnn_content(gnn_content: str) -> dict:
    """
    Parses the string content of a GNN file into a dictionary of sections.
    Each section's content is a list of non-empty, non-comment lines.
    Parses ## TensorDefinitions section specifically.
    """
    parsed_data: Dict[str, Union[List[str], Dict[str, Any]]] = {}  # Explicitly typed
    current_section_name: Optional[str] = None

    section_header_pattern = re.compile(r"^##\s*([^#\n]+?)\s*(?:#.*)?$")

    lines = gnn_content.splitlines()

    for line_number, line_content in enumerate(lines):
        stripped_line = line_content.strip()

        header_match = section_header_pattern.match(stripped_line)
        if header_match:
            section_title_match = header_match.group(1)
            if section_title_match is not None:
                current_section_name = section_title_match.strip().replace(" ", "")
                if current_section_name not in parsed_data:
                    # Special handling for TensorDefinitions to store structured data
                    if current_section_name == "TensorDefinitions":
                        parsed_data[current_section_name] = {}  # Always init as dict
                    else:
                        parsed_data[current_section_name] = []
                logger.debug(
                    f"Found section: {current_section_name} at line {line_number + 1}"
                )
            else:
                logger.warning(
                    f"Matched a section header but failed to extract title at line {line_number + 1}: '{stripped_line}'"
                )
                current_section_name = None
            continue

        if current_section_name and stripped_line and not stripped_line.startswith("#"):
            if current_section_name == "TensorDefinitions":
                # Ensure this section is a dictionary before assigning
                if not isinstance(parsed_data.get(current_section_name), dict):
                    logger.error(
                        "Section 'TensorDefinitions' was not initialized as a dict. This is a bug."
                    )
                    # Force it to be a dict to prevent further errors, though data might be lost
                    parsed_data[current_section_name] = {}

                target_dict_for_tensor_defs = parsed_data[current_section_name]
                # Now we are sure target_dict_for_tensor_defs is a Dict an can perform assignment.
                # However, the type checker might still complain because parsed_data[current_section_name] is a Union.
                # A cast or more refined type structure might be needed if this persists.
                if isinstance(target_dict_for_tensor_defs, dict):
                    parts = [p.strip() for p in stripped_line.split("|")]
                    if len(parts) == 4:
                        box_name, dom_spec, cod_spec, init_str_raw = parts  # Renamed

                        init_str_for_json_parse = init_str_raw.strip()
                        potential_json_literal = False

                        # Check if it's a double-quoted string that might contain JSON
                        if init_str_for_json_parse.startswith(
                            '"'
                        ) and init_str_for_json_parse.endswith('"'):
                            inner_str = init_str_for_json_parse[1:-1]
                            # Check if the inner content looks like a JSON array or object
                            if (
                                inner_str.startswith("[") and inner_str.endswith("]")
                            ) or (
                                inner_str.startswith("{") and inner_str.endswith("}")
                            ):
                                init_str_for_json_parse = (
                                    inner_str  # Use the unquoted inner string
                                )
                                potential_json_literal = True
                        # Check if it's an unquoted string that already looks like JSON
                        elif (
                            init_str_for_json_parse.startswith("[")
                            and init_str_for_json_parse.endswith("]")
                        ) or (
                            init_str_for_json_parse.startswith("{")
                            and init_str_for_json_parse.endswith("}")
                        ):
                            potential_json_literal = True

                        # Attempt to parse as JSON only if it's a potential literal and not "load:"
                        if (
                            potential_json_literal
                            and not init_str_raw.strip().startswith("load:")
                        ):
                            try:
                                initializer_from_json = json.loads(
                                    init_str_for_json_parse
                                )
                                logger.debug(
                                    f"TensorDef: Parsed init_str for '{box_name}' with json.loads. Type: {type(initializer_from_json)}, Value: {str(initializer_from_json)[:200]}"
                                )
                                initializer = initializer_from_json
                            except json.JSONDecodeError as e_json:
                                logger.warning(
                                    f"TensorDef: json.loads failed for '{box_name}' after attempting to unquote/prepare. Error: {e_json}. init_str_for_json_parse was: {init_str_for_json_parse}"
                                )
                                initializer = (
                                    init_str_raw.strip()
                                )  # Recovery to stripped raw string
                        else:
                            # Not a JSON literal (e.g. "load:...", "random_normal", or a simple string name for a function)
                            logger.debug(
                                f"TensorDef: Initializer for '{box_name}' not treated as direct JSON: {init_str_raw}"
                            )
                            initializer = init_str_raw.strip()  # Store it stripped

                        target_dict_for_tensor_defs[box_name] = {
                            "dom_spec": dom_spec,
                            "cod_spec": cod_spec,
                            "initializer": initializer,
                        }
                    else:
                        logger.warning(
                            f"Could not parse TensorDefinitions line {line_number + 1}: '{stripped_line}'. "
                            "Expected format: BoxName | DomDimsStr | CodDimsStr | InitializerStr"
                        )
                else:
                    # This case should ideally not be reached due to prior checks and initialization
                    logger.error(
                        f"Type error: 'TensorDefinitions' section resolved to non-dict type before assignment attempt at line {line_number + 1}."
                    )

            elif current_section_name in parsed_data:
                section_content = parsed_data[current_section_name]
                if isinstance(section_content, list):
                    section_content.append(stripped_line)
                else:  # Should not happen if initialized correctly
                    logger.error(
                        f"Section '{current_section_name}' is not a list. Line: '{stripped_line}'"
                    )
            else:
                logger.warning(
                    f"Attempting to add line to section '{current_section_name}' which was not "
                    f"initialized in parsed_data. Line: '{stripped_line}'. This may indicate a parsing logic error."
                )

    if logger.isEnabledFor(logging.DEBUG):
        for section, content_lines_or_dict in parsed_data.items():
            if isinstance(content_lines_or_dict, list):
                logger.debug(
                    f"  Section '{section}' has {len(content_lines_or_dict)} relevant lines."
                )
                if content_lines_or_dict:
                    logger.debug(
                        f"    First line of '{section}': '{content_lines_or_dict[0]}'"
                    )
            elif isinstance(content_lines_or_dict, dict):
                logger.debug(
                    f"  Section '{section}' has {len(content_lines_or_dict)} definitions."
                )
                if content_lines_or_dict:
                    first_key = next(iter(content_lines_or_dict))
                    logger.debug(
                        f"    First definition in '{section}': '{first_key}': {content_lines_or_dict[first_key]}"
                    )

    # logger.debug(f"Final parsed_data before return: {json.dumps(parsed_data, indent=2)[:1000]}")
    return parsed_data
