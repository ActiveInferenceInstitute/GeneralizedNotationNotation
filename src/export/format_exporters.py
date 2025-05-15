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

def _strip_comments_from_multiline_str(m_str: str) -> str:
    """Removes Python-style comments from a multi-line string."""
    lines = []
    for line in m_str.splitlines():
        stripped_line = line.split('#', 1)[0].rstrip()
        lines.append(stripped_line)
    # Join and then remove lines that became empty AFTER comment stripping and rstrip
    # but preserve structure for multiline arrays that might have legitimate empty lines (though uncommon)
    # For ast.literal_eval, truly empty lines within a list/tuple definition are often problematic anyway
    # So, filtering them out is usually safer if they are not part of string literals.
    # A simple join and then re-strip should be fine for ast.literal_eval
    return "\n".join(lines).strip() # Final strip to remove leading/trailing empty lines from the whole block

def _parse_matrix_string(matrix_str: str) -> Any:
    """Safely parses a string representation of a matrix after stripping comments."""
    
    processed_str = _strip_comments_from_multiline_str(matrix_str)
    # After stripping comments, processed_str might be empty or just whitespace
    if not processed_str:
        logger.debug(f"Matrix string was empty after comment stripping (original: '{matrix_str}')")
        return matrix_str # Or perhaps None, or an empty list, depending on desired behavior

    # Heuristic to convert GNN's common {{...}} or {(...)} for parameterization
    # into valid Python literal strings, typically aiming for list-of-lists or list-of-tuples.
    # This should happen AFTER comment stripping.
    if processed_str.startswith("{") and processed_str.endswith("}"):
        inner_content = processed_str[1:-1].strip()
        # If it looks like a dict, leave it as is for ast.literal_eval
        if ':' in inner_content and not (inner_content.startswith("(") and inner_content.endswith(")")):
            pass # Likely a dictionary, ast.literal_eval handles dicts with {}
        else:
            # Otherwise, assume GNN's { } means a list-like structure (set or list of items/tuples)
            # Convert to [ ] for ast.literal_eval to parse as a list.
            processed_str = "[" + inner_content + "]"

    try:
        parsed_value = ast.literal_eval(processed_str)
        
        def convert_structure(item):
            if isinstance(item, set):
                # Convert sets to sorted lists for deterministic output
                try:
                    return sorted(list(item))
                except TypeError:
                    # Cannot sort if items are of mixed uncomparable types (e.g. int and tuple)
                    return list(item)
            elif isinstance(item, list):
                return [convert_structure(x) for x in item]
            elif isinstance(item, tuple):
                return tuple(convert_structure(x) for x in item)
            elif isinstance(item, dict):
                return {k: convert_structure(v) for k, v in item.items()}
            return item

        parsed_value = convert_structure(parsed_value)
        
        # If the original GNN was like D_f0={(1.0,0.0,0.0)} and became [[(1.0,0.0,0.0)]] due to {[()]} heuristic,
        # and only contains one element that is a list/tuple, unwrap it.
        if isinstance(parsed_value, list) and len(parsed_value) == 1 and isinstance(parsed_value[0], (list,tuple)) and processed_str.startswith('[(') and processed_str.endswith(')]'):
            if processed_str.count('(') == 1 and processed_str.count(')') == 1 : # Check if it was a single tuple in original like {(...)}
                 parsed_value = list(parsed_value[0]) # Convert the inner tuple to list

        logger.debug(f"Parsed matrix string (original: \'{matrix_str}\') to (processed for eval: \'{processed_str}\'): {parsed_value}")
        return parsed_value
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        logger.warning(f"Error parsing matrix string with ast.literal_eval (original: \'{matrix_str}\', processed for eval: \'{processed_str}\'). Error: {e}. Returning as raw string.")
        return matrix_str

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
    Parses a line describing a transition or connection.
    Example: s1 -> s2 : probability=0.8, action="A1" label="Transition X"
    Also handles: (s1, s2) -> (s3, s4)
                  s1 > s2
                  s1 - s2 (simple link)
    Prime characters like s' are supported in IDs.
    """
    # Non-verbose, single-line raw string regex pattern
    pattern = r"^\s*(\(?[a-zA-Z0-9_,'\s]+\)?|[a-zA-Z0-9_']+)\s*([-><]+|-)\s*(\(?[a-zA-Z0-9_,'\s]+\)?|[a-zA-Z0-9_']+)\s*(?::\s*(.*))?$"
    match = re.match(pattern, line)

    if not match:
        logger.debug(f"Could not parse transition/connection line: {line}")
        return None

    source_str, operator, target_str, attrs_str = match.groups()

    def clean_variable_list_str(s: str) -> List[str]:
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1] # Remove parentheses
        return [v.strip() for v in s.split(',') if v.strip()]

    sources = clean_variable_list_str(source_str)
    targets = clean_variable_list_str(target_str)
    
    attributes = {}
    if attrs_str:
        # Regex to find key="value" or key='value' or key=bare_value
        # Handles escaped quotes within quoted values.
        attr_pairs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\\s*=\\s*("[^"\\\\]*(?:\\\\.[^"\\\\]*)*"|\'[^\'\\\\]*(?:\\\\.[^\'\\\\]*)*\'|[^{},\\s]+)', attrs_str)
        for key_attr, value_attr in attr_pairs: # Renamed to avoid conflict
            key_attr = key_attr.strip()
            value_attr = value_attr.strip()
            # Attempt to evaluate if it looks like a string literal, to unescape and convert
            if (value_attr.startswith('"') and value_attr.endswith('"')) or \
               (value_attr.startswith("'") and value_attr.endswith("'")): # Corrected
                try:
                    value_attr = ast.literal_eval(value_attr)
                except Exception: # Broad exception to catch any ast.literal_eval issues
                    logger.warning(f"Could not ast.literal_eval attribute value '{value_attr}' for key '{key_attr}'. Keeping as raw quoted string.")
            attributes[key_attr] = value_attr
        
    return {"sources": sources, "operator": operator.strip(), "targets": targets, "attributes": attributes}

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
    for line in section_content.strip().split('\\n'):
        line_stripped_comments = line.split('#', 1)[0].strip() # Remove comments before parsing
        if not line_stripped_comments: # Skip empty or comment-only lines
            continue
        
        match = re.match(r"([\\w_]+):\\s*(\\[.*?\\])", line_stripped_comments) # Regex for key: [list]
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
        elif ':' in line_stripped_comments: # General key: value fallback
            key_part, value_part = line_stripped_comments.split(":", 1)
            key = key_part.strip()
            value = value_part.strip() 
            data[key] = value # Store as string, could attempt _parse_matrix_string if values can be complex
            logger.debug(f"  Parsed ModelParameter (as string): {key} = {value}")
        else:
            logger.debug(f"  Skipping malformed line in ModelParameters: {line}")
    return data

def _parse_initial_parameterization_section(section_content: str) -> dict:
    """
    Parses the InitialParameterization section.
    Keys are parameter names (e.g., A_m0, D_f1).
    Values are GNN matrix strings which are parsed into Python objects (lists/tuples).
    Handles multi-line values for a single parameter.
    """
    data = {}
    current_key: Optional[str] = None
    current_value_lines: List[str] = []
    
    for line_raw in section_content.split('\n'):
        # A new parameter key is expected to be at the start of a line (ignoring whitespace)
        # and not be part of a comment.
        stripped_line_for_key_check = line_raw.lstrip()
        
        is_new_key_line = False
        if not stripped_line_for_key_check.startswith('#') and '=' in stripped_line_for_key_check:
            # Try to match "key = value" where key is simple alphanumeric.
            # This assumes the first '=' on such a line is the delimiter.
            match = re.match(r"^([a-zA-Z0-9_]+)\\s*=\\s*(.*)", stripped_line_for_key_check)
            if match:
                is_new_key_line = True
        
        if is_new_key_line and match: # Confirm match is not None
            # If there was a previous key, process its collected value lines
            if current_key is not None and current_value_lines:
                val_str_collected = "\n".join(current_value_lines).strip() # Strip whole block
                if val_str_collected: # only parse if non-empty after stripping
                    data[current_key] = _parse_matrix_string(val_str_collected)
                else:
                    data[current_key] = "" # Or some indicator of empty value if appropriate
                    logger.debug(f"Collected value for '{current_key}' was empty after stripping comments.")
            
            current_key = match.group(1).strip()
            initial_value_part = match.group(2) # This is the rest of the line after "key ="
            current_value_lines = [initial_value_part.strip()] # Start new value collection, strip this first part
        elif current_key is not None:
            # This line is a continuation of the previous key's value
            # We append the raw line to preserve its original content (including leading whitespace)
            # as _parse_matrix_string will handle comment stripping for the whole block later.
            current_value_lines.append(line_raw) 
        else:
            # This line is not a new key and there's no current_key being processed.
            # It might be a full-line comment or a malformed line at the start of the section.
            if not line_raw.strip().startswith('#') and line_raw.strip():
                 logger.debug(f"Skipping orphan/malformed line at start/between params in InitialParameterization: '{line_raw}'")
    
    # Process the last collected parameter after the loop ends
    if current_key is not None and current_value_lines:
        val_str_collected = "\n".join(current_value_lines).strip()
        if val_str_collected:
            data[current_key] = _parse_matrix_string(val_str_collected)
        else:
            data[current_key] = ""
            logger.debug(f"Collected value for '{current_key}' (last param) was empty after stripping comments.")
            
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