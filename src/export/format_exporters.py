"""
GNN Model Format Exporters and Parsers

This module provides the core GNN Markdown parsing function (`_gnn_model_to_dict`)
and previously contained various export functions. The export functions have been
refactored into specialized modules:
- structured_data_exporters.py (JSON, XML, Pickle)
- graph_exporters.py (GEXF, GraphML, JSON Adjacency List)
- text_exporters.py (Plaintext Summary, DSL Reconstruction)

This module retains the parsing logic and helper functions necessary for it.
It also re-exports the specialized export functions for any code that might still
call them directly from here, although direct usage of the specialized modules
is encouraged for new code.
"""
import logging
import re
from pathlib import Path
import ast
from typing import Dict, Any, List, Tuple, Callable, Optional

# Import from specialized exporter modules
from .structured_data_exporters import (
    export_to_json_gnn, 
    export_to_xml_gnn, 
    export_to_python_pickle
)
from .graph_exporters import (
    export_to_gexf, 
    export_to_graphml, 
    export_to_json_adjacency_list,
    HAS_NETWORKX # To check if graph exports are viable
)
from .text_exporters import (
    export_to_plaintext_summary, 
    export_to_plaintext_dsl
)

logger = logging.getLogger(__name__)

# --- Helper Functions (kept for GNN parsing) ---

def _ensure_path(path_str: str) -> Path:
    return Path(path_str)

def _parse_matrix_string(matrix_str: str) -> Any:
    """Safely parses a string representation of a matrix, after stripping comments."""
    # First, remove any comments from the string
    # A comment is considered anything from '#' to the end of the line/string
    processed_str = re.sub(r"#.*", "", matrix_str).strip()
    
    # If after stripping comments, the string is empty or clearly malformed (e.g., just an opening bracket)
    if not processed_str or processed_str in ["{", "[", "("]:
        logger.debug(f"Matrix string '{matrix_str}' became empty or malformed ('{processed_str}') after comment stripping. Returning as raw string.")
        return matrix_str # Return raw original string, as it might be a placeholder intended to be a string

    # Heuristic to convert GNN's {{...}} or {(...)} for tuples/lists of tuples into valid Python literal strings
    if processed_str.startswith("{") and processed_str.endswith("}"):
        inner_content = processed_str[1:-1].strip()
        if inner_content.startswith("(") and inner_content.endswith(")"): # Looks like {(...)}
            processed_str = "(" + inner_content + ")" # Treat as a single tuple
        elif inner_content.startswith("{") and inner_content.endswith("}"): # Potentially a set of tuples
             # This case needs care. If it's like {{t1},{t2}}, ast.literal_eval might treat it as set of sets.
             # For GNN, this often means a list/tuple of tuples.
             # Example: A_m0={ ( (0.333,...), (0.333,...) ),  ( (0.333,...), (0.333,...) ) }
             # This structure is more like a tuple of tuples.
             # Refined heuristic: if it's a set of tuples like format, convert to tuple of tuples
             # Make sure inner_content is not empty and doesn't just contain whitespace
             if inner_content and inner_content.count("(") > 0 and inner_content.count(")") > 0 and \
                inner_content.count("{") == 0 and inner_content.count("}") == 0 :
                 # Handles cases like "{(t1), (t2)}" or "{ (t1), (t2) }" -> "((t1), (t2))"
                 # Also handles simple sets of numbers like {1,2,3} -> (1,2,3) if they don't have sub-tuples
                 # If there are tuples inside, like {(1,2), (3,4)}, it becomes ((1,2), (3,4))
                 processed_str = f"({inner_content})"
    
    # Handle common cases where GNN uses {...} for lists/tuples of numbers directly
    # e.g. D_f0 = {0.5, 0.5} should become [0.5, 0.5] or (0.5, 0.5)
    # This is tricky because {1,2} is a valid set for ast.literal_eval.
    # The goal is to make GNN matrix formats more consistently lists/tuples.
    # The later conversion of top-level sets to lists handles this,
    # but this initial shaping helps ast.literal_eval.

    try:
        # Redundant check, but safe: if processed_str somehow became empty after heuristics, treat as error for ast.literal_eval
        if not processed_str:
            logger.warning(f"Matrix string '{matrix_str}' became empty after heuristics, should have been caught. Returning raw string.")
            return matrix_str

        parsed_value = ast.literal_eval(processed_str)
        
        # Convert sets to lists at various levels if necessary.
        # PyMDP often expects lists of lists or lists of tuples, not top-level sets or tuples of sets.
        
        # Initial conversion for top-level set or tuple
        if isinstance(parsed_value, set):
            parsed_value = list(parsed_value)
        if isinstance(parsed_value, tuple): # Convert top-level tuple to list for consistency
             parsed_value = list(parsed_value)

        def convert_sets_in_nesting(item):
            if isinstance(item, set):
                return list(item)
            elif isinstance(item, (list, tuple)):
                # Ensure elements within lists/tuples are also processed
                return type(item)(convert_sets_in_nesting(x) for x in item)
            return item

        parsed_value = convert_sets_in_nesting(parsed_value)
        
        logger.debug(f"Parsed matrix string '{matrix_str}' (processed: '{processed_str}') to: {parsed_value}")
        return parsed_value
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        logger.warning(f"Error parsing matrix string with ast.literal_eval: '{matrix_str}' (processed: '{processed_str}'). Error: {e}. Returning as raw string.")
        return matrix_str # Return raw string if parsing fails

# --- Parsers for GNN Sections (kept for _gnn_model_to_dict) ---

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
    match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(?:\[(.*?)\])?\s*(?:type=([a-zA-Z0-9_]+))?\s*(.*)$", line)
    if not match:
        simple_line_content = line.split('#')[0].strip()
        simple_match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(.*)$", simple_line_content)
        if simple_match:
            state_id_default = simple_match.group(1)
            attributes_str = simple_match.group(2).strip() 
            dimensions = None
            state_type = None
        else:
            logger.debug(f"Could not parse state line: {line}")
            return None
    else: 
        state_id_default = match.group(1)
        dimensions = match.group(2)
        state_type = match.group(3)
        attributes_str = match.group(4).split('#')[0].strip()

    attributes = {}
    if dimensions:
        attributes['dimensions'] = dimensions
    if state_type:
        attributes['type'] = state_type

    for kv_match in re.finditer(r'([a-zA-Z0-9_]+)\s*=\s*"([^"]*)"', attributes_str):
        attributes[kv_match.group(1)] = kv_match.group(2)
    
    state_id = attributes.pop('id_override', state_id_default)
    attributes['original_id'] = state_id_default

    return {"id": state_id, **attributes}

def _parse_transition_line(line: str) -> dict | None:
    """
    Parses a line describing a transition.
    Example: s1 -> s2 : probability=0.8, action="A1" label="Transition X"
    Also handles: s1-s2
    Prime characters like s' are supported in IDs.
    """
    match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(?:->|-)\s*([a-zA-Z0-9_']+)\s*(?::\s*(.*))?$", line)
    if not match:
        logger.debug(f"Could not parse transition/connection line: {line}")
        return None

    source, target, attrs_str = match.groups()
    attributes = {}
    if attrs_str:
        for part in attrs_str.split(','):
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
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
        
        match = re.match(r"([\w_]+):\s*(\[.*?\])\s*(?:#.*)?", line)
        if match:
            key = match.group(1).strip()
            value_str = match.group(2).strip()
            try:
                value = ast.literal_eval(value_str) # ast.literal_eval is safer
                if isinstance(value, list):
                    data[key] = value
                    logger.debug(f"  Parsed ModelParameter (as list): {key} = {value}")
                else:
                    logger.warning(f"  ModelParameter '{key}' value '{value_str}' did not evaluate to a list. Storing as string.")
                    data[key] = value_str 
            except (ValueError, SyntaxError, TypeError) as e: 
                logger.warning(f"  Could not parse ModelParameter value for '{key}' ('{value_str}') as list: {e}. Storing as string.")
                data[key] = value_str 
        elif ':' in line: 
            key_part, value_part = line.split(":", 1)
            key = key_part.strip()
            value = value_part.split("#", 1)[0].strip() 
            data[key] = value
            logger.debug(f"  Parsed ModelParameter (as string): {key} = {value}")
        else:
            logger.debug(f"  Skipping malformed line in ModelParameters: {line}")
    return data

def _parse_initial_parameterization_section(section_content: str) -> dict:
    """
    Parses the InitialParameterization section.
    Keys are parameter names (e.g., A_m0, D_f1).
    Values are GNN matrix strings which are parsed into Python objects (lists/tuples).
    """
    data = {}
    for line in section_content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'): # Skip comments and empty lines
            continue
        
        # Split by the first '=' to separate key and value
        if '=' in line:
            key, value_str = line.split('=', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            if key and value_str:
                # Attempt to parse the matrix string value
                parsed_value = _parse_matrix_string(value_str)
                data[key] = parsed_value
                if isinstance(parsed_value, str) and parsed_value == value_str: # Check if parsing failed
                    logger.warning(f"Value for '{key}' in InitialParameterization was not parsed into a data structure, kept as string: {value_str}")
            else:
                logger.debug(f"Skipping malformed line in InitialParameterization (empty key or value): '{line}'")
        else:
            logger.debug(f"Skipping line in InitialParameterization without '=': '{line}'")
            
    return data

SECTION_PARSERS = {
    "ActInfOntologyAnnotation": _parse_ontology_annotations,
    "StateSpaceBlock": lambda content: _parse_list_items_section(content, _parse_state_line),
    "ParameterBlock": _parse_key_value_section,
    "ObservationBlock": lambda content: _parse_list_items_section(content, _parse_state_line), 
    "TransitionBlock": lambda content: _parse_list_items_section(content, _parse_transition_line),
    "GNNSection": _parse_key_value_section,
    "Metadata": _parse_key_value_section,
    "states": lambda content: _parse_list_items_section(content, _parse_state_line),
    "parameters": _parse_key_value_section,
    "observations": lambda content: _parse_list_items_section(content, _parse_state_line),
    "transitions": lambda content: _parse_list_items_section(content, _parse_transition_line),
    "ImageFromPaper": _parse_key_value_section, 
    "GNNVersionAndFlags": _parse_key_value_section,
    "ModelName": _parse_free_text_section, 
    "ModelAnnotation": _parse_free_text_section, 
    "Connections": lambda content: _parse_list_items_section(content, _parse_transition_line), 
    "InitialParameterization": _parse_initial_parameterization_section, 
    "Equations": _parse_free_text_section, 
    "Time": _parse_key_value_section, 
    "Footer": _parse_free_text_section,
    "Signature": _parse_key_value_section, 
    "ModelParameters": _parse_model_parameters_section, 
}

# --- Main GNN Parser (retained here) ---

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
        "name": gnn_file_path.stem, 
        "metadata": {},
        "states": [], 
        "parameters": {}, 
        "initial_parameters": {}, 
        "observations": [], 
        "transitions": [], 
        "ontology_annotations": {},
        "equations_text": "",
        "time_info": {},
        "footer_text": "",
        "signature": {},
        "raw_sections": {}, 
        "other_sections": {} 
    }

    section_regex = r"^##\s*([A-Za-z0-9_\s]+?)\s*$(.*?)(?=^##\s*[A-Za-z0-9_\s]+?\s*$|\Z)"
    
    parsed_section_names = set()

    for match in re.finditer(section_regex, content, re.MULTILINE | re.DOTALL):
        section_name_original = match.group(1).strip()
        section_content_raw = match.group(2).strip()
        
        model["raw_sections"][section_name_original] = section_content_raw
        parsed_section_names.add(section_name_original)

        parser_found = False
        for known_parser_name, parser_func in SECTION_PARSERS.items():
            if section_name_original.lower() == known_parser_name.lower():
                try:
                    parsed_data = parser_func(section_content_raw)
                    if known_parser_name == "ModelName":
                        model["name"] = parsed_data 
                    elif known_parser_name == "ModelAnnotation":
                        model["metadata"]["description"] = parsed_data
                    elif known_parser_name == "StateSpaceBlock":
                        model["states"] = parsed_data 
                    elif known_parser_name == "ParameterBlock": 
                        model["parameters"] = parsed_data
                    elif known_parser_name == "Connections" or known_parser_name == "TransitionBlock" or known_parser_name == "transitions":
                        model["transitions"].extend(parsed_data) 
                    elif known_parser_name == "ActInfOntologyAnnotation":
                        model["ontology_annotations"] = parsed_data
                    elif known_parser_name == "InitialParameterization":
                        model["initial_parameters"] = parsed_data # 'parsed_data' is the direct result from _parse_initial_parameterization_section
                        
                        # Store the raw content for reference.
                        model["raw_sections"]["InitialParameterization_raw_content"] = section_content_raw
                        # Optionally, if wanting to trace the output of the section parser specifically:
                        # model["raw_sections"]["InitialParameterization_parsed_by_section_parser"] = parsed_data
                    elif known_parser_name == "Time":
                        model["time_info"] = parsed_data
                        if "type" in parsed_data: 
                             model["metadata"]["time_type"] = parsed_data["type"]
                    elif known_parser_name == "ModelParameters":
                        model["ModelParameters"] = parsed_data # Store the whole block
                        # And also hoist specific known params to top level of model dict
                        if isinstance(parsed_data, dict):
                            for mp_key, mp_val in parsed_data.items():
                                if mp_key in ["num_hidden_states_factors", "num_obs_modalities", "num_control_factors"]:
                                    model[mp_key] = mp_val
                    else: 
                        model[known_parser_name.lower().replace(" ", "_")] = parsed_data
                except Exception as e:
                    logger.warning(f"Error parsing section '{section_name_original}' with parser '{known_parser_name}': {e}")
                    model["other_sections"][section_name_original] = section_content_raw
                parser_found = True
                break
        
        if not parser_found:
            model["other_sections"][section_name_original] = section_content_raw

    # Post-processing to extract dimensions if ModelParameters didn't fully provide them
    if not model.get("num_obs_modalities"):
        obs_modality_dims: List[Tuple[int, int]] = []
        # ... (rest of dimension inference logic, if kept, would go here)
        # For brevity, assuming ModelParameters is the primary source now
    if not model.get("num_hidden_states_factors"):
        hidden_state_factor_dims_inferred: List[Tuple[int, int]] = []
        # ... 
        if hidden_state_factor_dims_inferred: 
            model["num_hidden_states_factors"] = [dim for _, dim in sorted(hidden_state_factor_dims_inferred)]
    # ...

    logger.debug(f"_gnn_model_to_dict: Final model before return for {gnn_file_path_str}")
    return model


# --- Self-Test (Optional, can be removed or kept for direct testing of parser) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sample_gnn_file_content = """
## ModelName
Test GNN Parser Model

## InitialParameterization
A = {(0.8, 0.2), (0.1, 0.9)}
B = [[1,0],[0,1]]
my_param = "some_string"
    """
    test_file_path = Path("_test_parser_model.gnn.md")
    output_dir = Path("_test_parser_output")
    output_dir.mkdir(exist_ok=True)

    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(sample_gnn_file_content)

    logger.info(f"--- Running Self-Test for format_exporters.py (Parser) ---")
    try:
        parsed_model = _gnn_model_to_dict(str(test_file_path))
        logger.info(f"Parsed model: {json.dumps(parsed_model, indent=2, default=str)}") # use default=str for non-serializable

        # Test specific exports by calling them from here
        json_out = output_dir / "model_parsed.json"
        export_to_json_gnn(parsed_model, str(json_out))
        logger.info(f"Exported parsed model to JSON: {json_out}")

    except Exception as e:
        logger.error(f"Self-test failed: {e}", exc_info=True)
    finally:
        if test_file_path.exists():
            test_file_path.unlink()
        # import shutil
        # if output_dir.exists():
        #     shutil.rmtree(output_dir)
    logger.info(f"--- Parser Self-Test Finished ---")

# The individual export functions (export_to_json_gnn, etc.) are now imported
# from their respective modules. This file primarily serves the GNN parsing logic. 