"""
GNN Model Format Exporters

This module provides functions to parse GNN Markdown files into a structured
dictionary representation and then export this representation to various formats,
including JSON, XML, plain text summaries, the original DSL, graph formats (GEXF, GraphML),
and Python pickles.
"""
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle
import logging
import re
from pathlib import Path
import ast
from typing import Dict, Any, List, Tuple, Callable, Optional

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None  # Handled in functions that require NetworkX

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _ensure_path(path_str: str) -> Path:
    return Path(path_str)

def _pretty_print_xml(element: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def _dict_to_xml(tag: str, d: dict | list | str | int | float | bool | None) -> ET.Element:
    """Recursively convert a Python dictionary or list to an XML ET.Element."""
    elem = ET.Element(tag)
    if isinstance(d, dict):
        for key, val in d.items():
            # Ensure keys are valid XML tag names
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
            if not safe_key or not safe_key[0].isalpha() and safe_key[0] != '_':
                safe_key = '_' + safe_key # Ensure it starts with a letter or underscore
            child = _dict_to_xml(safe_key, val)
            elem.append(child)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            item_tag = f"{tag}_item" # Generic item tag for list elements
            child = _dict_to_xml(item_tag, item)
            elem.append(child)
    elif d is None:
        elem.text = "" # Or handle as an attribute xsi:nil="true" if needed
    else:
        elem.text = str(d)
    return elem

def _parse_matrix_string(matrix_str: str) -> Any:
    """Safely parses a string representation of a matrix."""
    processed_str = matrix_str.strip()
    # GNN examples like A={(0.7,0.3),(0.4,0.6)} or D={(0.5),(0.5)}
    # These look like sets of tuples. ast.literal_eval can parse sets.
    # For pymdp, we usually want lists of lists or lists of tuples, or np.arrays.
    # Aim to convert to a tuple of tuples for immutable matrices.
    if processed_str.startswith("{") and processed_str.endswith("}") and \
       not (processed_str.startswith("{{") or processed_str.startswith("{\"") or processed_str.startswith("{[")): 
        # Avoid misinterpreting JSON-like dict strings as sets of tuples.
        # Check for presence of tuples inside to differentiate from a simple set of numbers.
        if '(' in processed_str[1:-1] and ')' in processed_str[1:-1]:
            # Convert GNN's { (tuple1), (tuple2) } to Python's ( (tuple1), (tuple2) )
            # Example: "{(0.7,0.3),(0.4,0.6)}" -> "((0.7,0.3),(0.4,0.6))"
            # Example: "{(0.5),(0.5)}" -> "((0.5),(0.5))"
            # This also handles D = {(0.5), (0.5)} -> ( (0.5), (0.5) ) -> tuple of two 1-element tuples.
            # And D = {(0.5, 0.5)} -> ((0.5, 0.5)), a tuple with one 2-element tuple.
            processed_str = "(" + processed_str[1:-1] + ")"
        # else: if it's like {0.5, 0.6}, ast.literal_eval will treat it as a set {0.5, 0.6}.
        # This is fine, as the renderer will then convert it to a list or array if needed.

    try:
        return ast.literal_eval(processed_str)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        print(f"Error parsing matrix string with ast.literal_eval: '{matrix_str}' (processed to: '{processed_str}'). Error: {e}")
        return None

# --- Parsers for GNN Sections ---

def _parse_free_text_section(section_content: str) -> str:
    """Parses section content as a block of free text."""
    return section_content.strip()

def _parse_key_value_section(section_content: str) -> dict:
    """Parses section content assuming key=value or key: value pairs per line."""
    data = {}
    for line in section_content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            data[key.strip()] = value.strip()
        elif ':' in line: # Alternative key-value
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    return data

def _parse_state_line(line: str) -> dict | None:
    """
    Parses a line describing a state.
    Example: s_t[2,1,type=float] name="Hidden State" id_override="state1"
    """
    # Regex to capture: id (allows prime), optional_dims, optional_type, and other key="value" pairs
    # Dimensions are matched by [non-bracket_content].
    # Prime characters like s' are supported in IDs.
    match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(?:\[(.*?)\])?\s*(?:type=([a-zA-Z0-9_]+))?\s*(.*)$", line)
    if not match:
        # Try a simpler regex for IDs without brackets or types, e.g. "x" or "y_prime"
        # Ensure to strip comments before this simpler match too.
        simple_line_content = line.split('#')[0].strip()
        simple_match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(.*)$", simple_line_content)
        if simple_match:
            state_id_default = simple_match.group(1)
            # attributes_str for simple match should only contain actual attributes, not just trailing spaces.
            attributes_str = simple_match.group(2).strip() 
            dimensions = None
            state_type = None
        else:
            logger.debug(f"Could not parse state line: {line}")
            return None
    else: # if the more complex regex matched
        state_id_default = match.group(1)
        dimensions = match.group(2)
        state_type = match.group(3)
        # Process attributes_str: remove comments, then parse key="value" pairs
        attributes_str = match.group(4).split('#')[0].strip()

    attributes = {}
    if dimensions:
        attributes['dimensions'] = dimensions
    if state_type:
        attributes['type'] = state_type

    # Parse additional key="value" attributes
    for kv_match in re.finditer(r'([a-zA-Z0-9_]+)\s*=\s*"([^"]*)"', attributes_str):
        attributes[kv_match.group(1)] = kv_match.group(2)
    
    # Use id_override if present, otherwise default
    state_id = attributes.pop('id_override', state_id_default)
    attributes['original_id'] = state_id_default # Keep original if overridden

    return {"id": state_id, **attributes}


def _parse_transition_line(line: str) -> dict | None:
    """
    Parses a line describing a transition.
    Example: s1 -> s2 : probability=0.8, action="A1" label="Transition X"
    Also handles: s1-s2
    Prime characters like s' are supported in IDs.
    """
    # Regex to capture source, target (allowing prime characters), and optional attributes, allowing for '->' or '-'
    match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(?:->|-)\s*([a-zA-Z0-9_']+)\s*(?::\s*(.*))?$", line)
    if not match:
        logger.debug(f"Could not parse transition/connection line: {line}")
        return None

    source, target, attrs_str = match.groups()
    attributes = {}
    if attrs_str:
        # Split by comma, but be careful with commas inside quotes (not handled here for simplicity)
        for part in attrs_str.split(','):
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                attributes[key] = value
    return {"source": source, "target": target, "attributes": attributes}

def _parse_list_items_section(section_content: str, item_parser: callable) -> list:
    """Parses lines in a section using a specific item_parser for each line."""
    items = []
    for line in section_content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parsed_item = item_parser(line)
        if parsed_item:
            items.append(parsed_item)
    return items

def _parse_ontology_annotations(section_content: str) -> dict:
    """Parses ActInfOntologyAnnotation section (key=value pairs)."""
    annotations = {}
    lines = section_content.strip().split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            parts = line.split('=', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            if key and value:
                annotations[key] = value
            else:
                logger.debug(f"Malformed line {i+1} in ontology annotation: '{line}' - skipping.")
        else:
            logger.debug(f"Line {i+1} in ontology annotation does not contain '=': '{line}' - skipping.")
    return annotations

def _parse_model_parameters_section(section_content: str) -> dict:
    """Parses ModelParameters section, converting list-like strings to Python lists."""
    data = {}
    for line in section_content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Regex to capture key and list-like value, allowing for comments
        match = re.match(r"([\w_]+):\s*(\[.*?\])\s*(?:#.*)?", line)
        if match:
            key = match.group(1).strip()
            value_str = match.group(2).strip()
            try:
                import ast
                value = ast.literal_eval(value_str)
                if isinstance(value, list):
                    data[key] = value
                    logger.debug(f"  Parsed ModelParameter (as list): {key} = {value}")
                else:
                    logger.warning(f"  ModelParameter '{key}' value '{value_str}' did not evaluate to a list. Storing as string.")
                    data[key] = value_str # Store as string if not a list
            except (ValueError, SyntaxError, TypeError) as e: # Added TypeError for safety
                logger.warning(f"  Could not parse ModelParameter value for '{key}' ('{value_str}') as list: {e}. Storing as string.")
                data[key] = value_str # Store as string on error
        elif ':' in line: # Fallback for general key:value not matching list format
            key_part, value_part = line.split(":", 1)
            key = key_part.strip()
            value = value_part.split("#", 1)[0].strip() # Remove comments
            data[key] = value
            logger.debug(f"  Parsed ModelParameter (as string): {key} = {value}")
        else:
            logger.debug(f"  Skipping malformed line in ModelParameters: {line}")
    return data

SECTION_PARSERS = {
    # Exact names first
    "ActInfOntologyAnnotation": _parse_ontology_annotations,
    "StateSpaceBlock": lambda content: _parse_list_items_section(content, _parse_state_line),
    "ParameterBlock": _parse_key_value_section,
    "ObservationBlock": lambda content: _parse_list_items_section(content, _parse_state_line), # Treat observations like states
    "TransitionBlock": lambda content: _parse_list_items_section(content, _parse_transition_line),
    "GNNSection": _parse_key_value_section,
    "Metadata": _parse_key_value_section,
    # Aliases or common variations (case-insensitive matching handled in main parser)
    "states": lambda content: _parse_list_items_section(content, _parse_state_line),
    "parameters": _parse_key_value_section,
    "observations": lambda content: _parse_list_items_section(content, _parse_state_line),
    "transitions": lambda content: _parse_list_items_section(content, _parse_transition_line),
    # Sections that were previously falling into 'other_sections'
    "ImageFromPaper": _parse_key_value_section, # Expects e.g., url="...", caption="..."
    "GNNVersionAndFlags": _parse_key_value_section,
    "ModelName": _parse_free_text_section, # Typically a single line of text
    "ModelAnnotation": _parse_free_text_section, # Can be multi-line descriptive text
    "Connections": lambda content: _parse_list_items_section(content, _parse_transition_line), # Assuming similar structure to transitions
    "InitialParameterization": _parse_key_value_section, # Or a more complex matrix parser if needed later
    "Equations": _parse_free_text_section, # Block of equations, possibly LaTeX
    "Time": _parse_key_value_section, # e.g., discrete="true", horizon="T"
    "Footer": _parse_free_text_section,
    "Signature": _parse_key_value_section, # e.g., author="...", date="..."
    "ModelParameters": _parse_model_parameters_section, # Added parser for ModelParameters
}

# --- Main GNN Parser ---

def _gnn_model_to_dict(gnn_file_path_str: str) -> dict:
    """
    Parses a GNN Markdown file into a structured dictionary.
    The GNN file is expected to have sections like ## SectionName.
    """
    gnn_file_path = Path(gnn_file_path_str)
    if not gnn_file_path.is_file():
        logger.error(f"GNN file not found: {gnn_file_path_str}")
        raise FileNotFoundError(f"GNN file not found: {gnn_file_path_str}")

    try:
        content = gnn_file_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading GNN file {gnn_file_path_str}: {e}")
        raise

    model = {
        "file_path": str(gnn_file_path),
        "name": gnn_file_path.stem, # Default name from file stem
        "metadata": {},
        "states": [], # Parsed from StateSpaceBlock or similar
        "parameters": {},
        "observations": [], # Potentially from a dedicated ObservationBlock
        "transitions": [], # Parsed from Connections or TransitionBlock
        "ontology_annotations": {},
        "equations_text": "",
        "time_info": {},
        "footer_text": "",
        "signature": {},
        "raw_sections": {}, # Store raw content of all sections found by ## Name
        "other_sections": {} # For sections not explicitly handled by specific parsers
    }

    # Regex to find ## SectionName and its content (non-greedy content capture)
    # This regex captures the section name and then everything until the next ## or end of file.
    section_regex = r"^##\s*([A-Za-z0-9_\s]+?)\s*$(.*?)(?=^##\s*[A-Za-z0-9_\s]+?\\s*$|\Z)"
    
    parsed_section_names = set()

    for match in re.finditer(section_regex, content, re.MULTILINE | re.DOTALL):
        section_name_original = match.group(1).strip() # Original name from ## header
        section_content_raw = match.group(2).strip()
        
        # Store raw section content
        model["raw_sections"][section_name_original] = section_content_raw
        parsed_section_names.add(section_name_original)

        # Attempt to parse using SECTION_PARSERS (case-insensitive matching for keys in SECTION_PARSERS)
        parser_found = False
        for known_parser_name, parser_func in SECTION_PARSERS.items():
            if section_name_original.lower() == known_parser_name.lower():
                try:
                    parsed_data = parser_func(section_content_raw)
                    # Integrate parsed data into the model dict based on parser type or section name
                    if known_parser_name == "ModelName":
                        model["name"] = parsed_data # Overrides default from filename
                    elif known_parser_name == "ModelAnnotation":
                        model["metadata"]["description"] = parsed_data
                    elif known_parser_name == "StateSpaceBlock":
                        logger.debug(f"Attempting to assign to model['states']. known_parser_name='{known_parser_name}'. Parsed data count: {len(parsed_data) if isinstance(parsed_data, list) else 'N/A (not a list)'}. Parsed data: {parsed_data}")
                        model["states"] = parsed_data # This should be a list of dicts
                    elif known_parser_name == "Connections" or known_parser_name == "TransitionBlock" or known_parser_name == "transitions":
                        logger.debug(f"Attempting to extend model['transitions']. known_parser_name='{known_parser_name}'. Parsed data count: {len(parsed_data) if isinstance(parsed_data, list) else 'N/A (not a list)'}. Parsed data: {parsed_data}")
                        model["transitions"].extend(parsed_data) # Accumulate if multiple transition sections
                    elif known_parser_name == "ActInfOntologyAnnotation":
                        model["ontology_annotations"] = parsed_data
                    elif known_parser_name == "InitialParameterization":
                        logger.debug(f"Attempting to assign to model['parameters']. known_parser_name='{known_parser_name}'. Parsed data count: {len(parsed_data) if isinstance(parsed_data, dict) else 'N/A (not a dict)'}. Parsed data: {parsed_data}")
                        # Store as key-value, but also try to parse matrix-like structures
                        model["parameters"] = parsed_data # _parse_key_value_section
                        # Attempt to parse matrices from the string values
                        parsed_matrices = {}
                        for param_name, param_val_str in model["parameters"].items():
                            try:
                                # Basic eval for tuple/list structures, BE CAREFUL WITH EVAL
                                # This is a security risk if GNN files come from untrusted sources.
                                # A safer parser would be needed for general untrusted inputs.
                                matrix_data = eval(param_val_str)
                                if isinstance(matrix_data, (list, tuple)):
                                    parsed_matrices[param_name] = matrix_data
                                else: # Keep as string if not list/tuple
                                    parsed_matrices[param_name] = param_val_str
                            except: # If eval fails, keep as string
                                parsed_matrices[param_name] = param_val_str
                        model["matrix_parameters"] = parsed_matrices # Store potentially eval'd matrices
                        model["InitialParameterization_raw"] = section_content_raw # Keep raw for other uses

                    elif known_parser_name == "Time":
                        model["time_info"] = parsed_data
                        if "type" in parsed_data: # e.g. type=Dynamic
                             model["metadata"]["time_type"] = parsed_data["type"]
                    # Add more specific integrations as needed
                    else: # General case: store under a key derived from parser_name or section_name_original
                        model[known_parser_name.lower().replace(" ", "_")] = parsed_data
                except Exception as e:
                    logger.warning(f"Error parsing section '{section_name_original}' with parser '{known_parser_name}': {e}")
                    model["other_sections"][section_name_original] = section_content_raw
                parser_found = True
                break
        
        if not parser_found:
            logger.debug(f"No specific parser for section '{section_name_original}'. Storing in 'other_sections'.")
            model["other_sections"][section_name_original] = section_content_raw

    # --- Post-processing to extract dimensions for PyMDP/RxInfer ---
    # This part relies on 'states' being populated by StateSpaceBlock parser,
    # and 'ontology_annotations' by ActInfOntologyAnnotation parser.

    # Initialize lists to store dimensions
    obs_modality_dims: List[Tuple[int, int]] = [] # (index, dimension)
    hidden_state_factor_dims: List[Tuple[int, int]] = [] # (index, dimension)
    control_factor_dims: List[Tuple[int, int]] = [] # (index, dimension)

    parsed_states_dict = {s['id']: s for s in model.get("states", [])} # For quick lookup
    ontology_annotations = model.get("ontology_annotations", {})

    # Iterate through GNN variables and their ontology annotations
    for var_id, ontology_term in ontology_annotations.items():
        state_info = parsed_states_dict.get(var_id)
        if not state_info:
            # This can happen if an ontology annotation refers to something not in StateSpaceBlock (e.g. a matrix name)
            # logger.debug(f"Variable '{var_id}' from ontology annotations not found in StateSpaceBlock. Skipping for dimension extraction.")
            continue

        dims_str = state_info.get("dimensions")
        if not dims_str:
            logger.debug(f"Variable '{var_id}' in StateSpaceBlock has no dimensions specified. Skipping for factor/modality extraction.")
            continue

        current_var_primary_dim = None
        try:
            # We typically need the first dimension (e.g., number of outcomes/states)
            dim_parts = [d.strip() for d in dims_str.split(',')]
            if dim_parts:
                current_var_primary_dim = int(dim_parts[0])
        except ValueError:
            logger.warning(f"Primary dimension for variable '{var_id}' ('{dims_str}') is not an integer. Cannot use for factor/modality count.")
            continue
        except Exception as e:
            logger.warning(f"Could not parse dimensions '{dims_str}' for variable '{var_id}': {e}")
            continue

        if current_var_primary_dim is None:
            continue

        # Check for Observation Modalities (e.g., ObservationModality0, ObservationModality1)
        obs_match = re.match(r"ObservationModality(\d+)", ontology_term)
        if obs_match:
            modality_index = int(obs_match.group(1))
            obs_modality_dims.append((modality_index, current_var_primary_dim))
            logger.debug(f"Found obs modality: '{var_id}' (index {modality_index}) with dim {current_var_primary_dim}")
            continue # Move to next ontology item

        # Check for Hidden State Factors (e.g., HiddenStateFactor0, HiddenStateFactor1)
        hs_match = re.match(r"HiddenStateFactor(\d+)", ontology_term)
        if hs_match:
            factor_index = int(hs_match.group(1))
            hidden_state_factor_dims.append((factor_index, current_var_primary_dim))
            logger.debug(f"Found hidden state factor: '{var_id}' (index {factor_index}) with dim {current_var_primary_dim}")
            continue

        # Check for Control State Factors (e.g., ControlStateFactor0, ControlFactor0)
        cs_match = re.match(r"(?:ControlStateFactor|ControlFactor)(\d+)", ontology_term)
        if cs_match:
            factor_index = int(cs_match.group(1))
            control_factor_dims.append((factor_index, current_var_primary_dim))
            logger.debug(f"Found control state factor: '{var_id}' (index {factor_index}) with dim {current_var_primary_dim}")
            continue
        
        # ADDED: Check for Policy Factors (e.g., PolicyVectorFactor0, PolicyFactor0)
        # Matches PolicyVectorFactor0, PolicyFactor0, PolicyVector0, Policy0
        policy_match_indexed = re.match(r"(?:PolicyVectorFactor|PolicyFactor|PolicyVector|Policy)(\d+)", ontology_term)
        if policy_match_indexed:
            factor_index = int(policy_match_indexed.group(1))
            # Ensure we don't double-add if already found by cs_match for the same index
            if not any(idx == factor_index for idx, _ in control_factor_dims):
                control_factor_dims.append((factor_index, current_var_primary_dim))
                logger.debug(f"Found policy factor: '{var_id}' (index {factor_index}) with dim {current_var_primary_dim} (via indexed Policy* pattern)")
            continue # Continue to next ontology item even if it was a duplicate add attempt

        # Handle simple "Observation", "HiddenState", "Control" for backward compatibility / simple models
        # These will be treated as modality/factor 0 if no indexed versions were found.
        if ontology_term == "Observation" and not obs_modality_dims: 
            obs_modality_dims.append((0, current_var_primary_dim))
            logger.debug(f"Found simple observation: '{var_id}' with dim {current_var_primary_dim}")
        elif ontology_term == "HiddenState" and not hidden_state_factor_dims:
            hidden_state_factor_dims.append((0, current_var_primary_dim))
            logger.debug(f"Found simple hidden state: '{var_id}' with dim {current_var_primary_dim}")
        elif (ontology_term == "Control" or ontology_term == "ControlState") and not control_factor_dims:
            control_factor_dims.append((0, current_var_primary_dim))
            logger.debug(f"Found simple control state: '{var_id}' with dim {current_var_primary_dim}")
        # ADDED: Handle simple "Policy" or "PolicyVector" if no indexed versions or simple Control were found
        elif (ontology_term == "Policy" or ontology_term == "PolicyVector") and not control_factor_dims:
            # This simple version is only added if control_factor_dims is still empty after checking indexed and simple Control terms
            control_factor_dims.append((0, current_var_primary_dim))
            logger.debug(f"Found simple policy: '{var_id}' with dim {current_var_primary_dim} (via simple Policy* pattern, as factor 0)")
        # No continue here, as a variable might match multiple simple fallbacks if logic is not exclusive,
        # though current structure (elif) makes it exclusive for simple terms.

    # Sort by index and extract dimensions
    model["num_obs_modalities"] = [dim for _, dim in sorted(obs_modality_dims)]
    model["num_hidden_states_factors"] = [dim for _, dim in sorted(hidden_state_factor_dims)]
    model["num_control_factors"] = [dim for _, dim in sorted(control_factor_dims)]

    logger.debug(f"Extracted num_obs_modalities: {model['num_obs_modalities']}")
    logger.debug(f"Extracted num_hidden_states_factors: {model['num_hidden_states_factors']}")
    logger.debug(f"Extracted num_control_factors: {model['num_control_factors']}")

    # Populate A_matrix, B_matrix etc. from model["matrix_parameters"] if they exist
    if "matrix_parameters" in model:
        for matrix_name_key in ["A", "B", "C", "D", "E"]: # Common PyMDP matrices
            if matrix_name_key in model["matrix_parameters"]:
                model[f"{matrix_name_key}_matrix"] = model["matrix_parameters"][matrix_name_key] # e.g. model["A_matrix"] = ...
                # For C and D, PyMDP often calls them vectors, but uses _matrix suffix in some contexts
                if matrix_name_key == "C":
                     model["C_vector"] = model["matrix_parameters"][matrix_name_key]
                if matrix_name_key == "D":
                     model["D_vector"] = model["matrix_parameters"][matrix_name_key]

    logger.debug(f"Final model states count: {len(model.get('states', []))}")
    logger.debug(f"Final model parameters count: {len(model.get('parameters', {}))}")
    logger.debug(f"Final model transitions count: {len(model.get('transitions', []))}")
    logger.debug(f"Final model states: {model.get('states')}")
    logger.debug(f"Final model parameters: {model.get('parameters')}")
    logger.debug(f"Final model transitions: {model.get('transitions')}")

    # --- Populate specific model parameters from parsed ModelParameters section ---
    # These will be top-level keys in the JSON if found in the ModelParameters section.
    # This makes them easily accessible for renderers and other downstream tools.
    model_params_parsed = parsed_data.get('ModelParameters', {})
    model["num_hidden_states_factors"] = model_params_parsed.get("num_hidden_states_factors", [])
    model["num_obs_modalities"] = model_params_parsed.get("num_obs_modalities", [])
    model["num_control_factors"] = model_params_parsed.get("num_control_factors", []) # Standardized key
    # Add any other expected parameters from ModelParameters here if needed
    # Example: model["policy_horizon"] = model_params_parsed.get("policy_horizon", None)

    logger.debug(f"Final model dictionary keys: {list(model.keys())}")
    logger.debug(f"num_hidden_states_factors from ModelParameters: {model.get('num_hidden_states_factors')}")
    logger.debug(f"num_obs_modalities from ModelParameters: {model.get('num_obs_modalities')}")
    logger.debug(f"num_control_factors from ModelParameters: {model.get('num_control_factors')}")

    # Infer dimensions if not explicitly provided (legacy or fallback)
    # This part might become redundant if ModelParameters is made mandatory for these values.
    if not model["num_hidden_states_factors"]:
        # Attempt to infer from StateSpaceBlock if still needed
        hidden_state_factor_dims = []
        # This inference logic needs to be robust and consider how states are defined in StateSpaceBlock
        # Example (simplified, needs actual parsing of state definitions):
        # for state_def in model.get("states", []):
        #     if state_def.get('id','').startswith('s_f'): # or other convention for hidden state factors
        #         dims_str = state_def.get('dimensions')
        #         if dims_str:
        #             try:
        #                 dim_val = int(dims_str.split(',')[0]) # Assuming first dim is factor size
        #                 hidden_state_factor_dims.append((state_def.get('id'), dim_val))
        #             except:
        #                 pass # Failed to parse dimension
        if hidden_state_factor_dims: # If inference yielded something
            model["num_hidden_states_factors"] = [dim for _, dim in sorted(hidden_state_factor_dims)]
            logger.debug(f"Inferred num_hidden_states_factors: {model['num_hidden_states_factors']}")

    # Similar inference for num_obs_modalities and num_control_factors if they are empty
    # and ModelParameters section didn't provide them. For now, we rely on ModelParameters.

    return model

# --- Export Functions ---

def export_to_json_gnn(gnn_model: dict, output_file_path: str):
    """Exports the GNN model dictionary to a JSON file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(gnn_model, f, indent=4, ensure_ascii=False)
        logger.debug(f"Successfully exported GNN model to JSON: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to JSON {output_file_path}: {e}", exc_info=True)
        raise

def export_to_xml_gnn(gnn_model: dict, output_file_path: str):
    """Exports the GNN model dictionary to an XML file."""
    try:
        root_element = _dict_to_xml('gnn_model', gnn_model)
        xml_string = _pretty_print_xml(root_element)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        logger.debug(f"Successfully exported GNN model to XML: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to XML {output_file_path}: {e}", exc_info=True)
        raise

def export_to_plaintext_summary(gnn_model: dict, output_file_path: str):
    """Exports a human-readable plain text summary of the GNN model."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"GNN Model Summary: {gnn_model.get('metadata', {}).get('name', 'N/A')}\\n")
            f.write(f"Source File: {gnn_model.get('file_path', 'N/A')}\\n\\n")

            f.write("Metadata:\\n")
            for k, v in gnn_model.get('metadata', {}).items():
                f.write(f"  {k}: {v}\\n")
            f.write("\\n")

            f.write(f"States ({len(gnn_model.get('states', []))}):\\n")
            for state in gnn_model.get('states', []):
                f.write(f"  - ID: {state.get('id')}")
                attrs = [f"{k}={v}" for k,v in state.items() if k!='id']
                if attrs: f.write(f" ({', '.join(attrs)})")
                f.write("\\n")
            f.write("\\n")

            f.write(f"Parameters ({len(gnn_model.get('parameters', {}))}):\\n")
            for k, v in gnn_model.get('parameters', {}).items():
                 f.write(f"  {k}: {v}\\n")
            f.write("\\n")
            
            f.write(f"Observations ({len(gnn_model.get('observations', []))}):\\n")
            for obs in gnn_model.get('observations', []):
                f.write(f"  - ID: {obs.get('id')}")
                attrs = [f"{k}={v}" for k,v in obs.items() if k!='id']
                if attrs: f.write(f" ({', '.join(attrs)})")
                f.write("\\n")
            f.write("\\n")

            f.write(f"Transitions ({len(gnn_model.get('transitions', []))}):\\n")
            for trans in gnn_model.get('transitions', []):
                attr_str = ", ".join([f"{k}={v}" for k, v in trans.get('attributes', {}).items()])
                f.write(f"  - {trans.get('source')} -> {trans.get('target')}")
                if attr_str: f.write(f" : {attr_str}")
                f.write("\\n")
            f.write("\\n")

            f.write(f"Ontology Annotations ({len(gnn_model.get('ontology_annotations', {}))}):\\n")
            for k, v in gnn_model.get('ontology_annotations', {}).items():
                f.write(f"  {k} = {v}\\n")
            f.write("\\n")

        logger.debug(f"Successfully exported GNN model to plain text summary: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to plain text summary {output_file_path}: {e}", exc_info=True)
        raise

def export_to_plaintext_dsl(gnn_model: dict, output_file_path: str):
    """
    Exports the GNN model back to a DSL-like format using the raw sections.
    This aims to reconstruct the original .gnn.md file structure.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Order of sections might matter for DSL, but raw_sections is a dict.
            # For now, write in the order they appear in the dict (Python 3.7+ insertion order).
            # A more robust solution might involve a predefined section order.
            for section_name, section_content in gnn_model.get('raw_sections', {}).items():
                f.write(f"## {section_name}\\n")
                f.write(f"{section_content}\\n\\n") # Add a newline after section content
        logger.debug(f"Successfully exported GNN model to plain text DSL: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to plain text DSL {output_file_path}: {e}", exc_info=True)
        raise

def _build_networkx_graph(gnn_model: dict) -> 'nx.DiGraph | None':
    """Helper to build a NetworkX graph from the GNN model."""
    if nx is None:
        logger.error("NetworkX library is not available. Cannot perform graph exports.")
        return None

    graph = nx.DiGraph()
    model_name = gnn_model.get('metadata', {}).get('name', 'GNN_Model')
    graph.graph['name'] = model_name

    # Add states as nodes
    for state_data in gnn_model.get('states', []):
        node_id = state_data.get('id')
        if node_id:
            attributes = {k: v for k, v in state_data.items() if k != 'id'}
            graph.add_node(node_id, **attributes)

    # Add observations as nodes (if distinct from states)
    # For simplicity, assuming observations might also be nodes if they have IDs
    for obs_data in gnn_model.get('observations', []):
        node_id = obs_data.get('id')
        if node_id and not graph.has_node(node_id): # Add if not already a state
            attributes = {k: v for k, v in obs_data.items() if k != 'id'}
            graph.add_node(node_id, **attributes)


    # Add transitions as edges
    for trans_data in gnn_model.get('transitions', []):
        source = trans_data.get('source')
        target = trans_data.get('target')
        if source and target:
            attributes = trans_data.get('attributes', {})
            # Ensure source and target nodes exist, add them if not (as simple nodes)
            if not graph.has_node(source): graph.add_node(source, label=source)
            if not graph.has_node(target): graph.add_node(target, label=target)
            graph.add_edge(source, target, **attributes)
            
    return graph

def export_to_gexf(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a GEXF file."""
    graph = _build_networkx_graph(gnn_model)
    if graph is None:
        # Error already logged by _build_networkx_graph
        raise ImportError("NetworkX not available, GEXF export failed.")
    try:
        nx.write_gexf(graph, output_file_path)
        logger.debug(f"Successfully exported GNN model to GEXF: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to GEXF {output_file_path}: {e}", exc_info=True)
        raise

def export_to_graphml(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a GraphML file."""
    graph = _build_networkx_graph(gnn_model)
    if graph is None:
        raise ImportError("NetworkX not available, GraphML export failed.")
    try:
        nx.write_graphml(graph, output_file_path)
        logger.debug(f"Successfully exported GNN model to GraphML: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to GraphML {output_file_path}: {e}", exc_info=True)
        raise

def export_to_json_adjacency_list(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a JSON adjacency list format."""
    graph = _build_networkx_graph(gnn_model)
    if graph is None:
        raise ImportError("NetworkX not available, JSON adjacency list export failed.")
    try:
        adj_data = nx.readwrite.json_graph.adjacency_data(graph)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(adj_data, f, indent=4, ensure_ascii=False)
        logger.debug(f"Successfully exported GNN model to JSON adjacency list: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to JSON adjacency list {output_file_path}: {e}", exc_info=True)
        raise

def export_to_python_pickle(gnn_model: dict, output_file_path: str):
    """Serializes the GNN model dictionary to a Python pickle file."""
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(gnn_model, f)
        logger.debug(f"Successfully exported GNN model to Python pickle: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to Python pickle {output_file_path}: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    # Basic self-test or example usage
    # To run this, you'd need a sample .gnn.md file.
    # Create a logger for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    sample_gnn_file_content = """
## GNNSection
name = MySampleModel
version = 1.0
description = A sample GNN model for testing exporters.

## StateSpaceBlock
s1 [type=hidden] name="State One"
s2 [type=observable] name="State Two"
s3 name="State Three"

## ParameterBlock
param_alpha = 0.5
param_beta = "default_value"

## ObservationBlock
obs1 [source=s1] name="Observation from s1"

## TransitionBlock
s1 -> s2 : probability=0.8, action="action_A"
s2 -> s3 : probability=1.0
s3 -> s1 : probability=0.2, label="Reset"

## ActInfOntologyAnnotation
s1 = HiddenStateAI
s2 = ObservableStateAI
param_alpha = LearningRate

## CustomUserSection
custom_key = custom_value
another_custom = 123
    """
    test_file_path = Path("sample_test_model.gnn.md")
    output_dir = Path("test_exports")
    output_dir.mkdir(exist_ok=True)

    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(sample_gnn_file_content)

    logger.info(f"--- Running Self-Test for format_exporters.py ---")
    try:
        parsed_model = _gnn_model_to_dict(str(test_file_path))
        logger.info(f"Parsed model: {json.dumps(parsed_model, indent=2)}")

        export_to_json_gnn(parsed_model, str(output_dir / "model.json"))
        export_to_xml_gnn(parsed_model, str(output_dir / "model.xml"))
        export_to_plaintext_summary(parsed_model, str(output_dir / "model_summary.txt"))
        export_to_plaintext_dsl(parsed_model, str(output_dir / "model_reconstructed.gnn.md"))
        export_to_python_pickle(parsed_model, str(output_dir / "model.pkl"))

        if nx:
            export_to_gexf(parsed_model, str(output_dir / "model.gexf"))
            export_to_graphml(parsed_model, str(output_dir / "model.graphml"))
            export_to_json_adjacency_list(parsed_model, str(output_dir / "model_adj.json"))
        else:
            logger.warning("Skipping graph exports as NetworkX is not available.")
            
        logger.info(f"Self-test exports completed. Check the '{output_dir.name}' directory.")

    except Exception as e:
        logger.error(f"Self-test failed: {e}", exc_info=True)
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()
        # Optionally clean up output_dir if desired, or leave for inspection
        # import shutil
        # if output_dir.exists():
        # shutil.rmtree(output_dir)
    logger.info(f"--- Self-Test Finished ---") 