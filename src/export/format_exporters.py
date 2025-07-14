"""
GNN Model Format Exporters and Parsers

This module provides the core GNN Markdown parsing function (`_gnn_model_to_dict`)
and a comprehensive set of functions to export the parsed GNN model into various
formats, including structured data (JSON, XML), graph formats (GEXF, GraphML),
and text-based representations (Summary, DSL).
"""
import logging
import re
from pathlib import Path
import ast
from typing import Dict, Any, List, Optional, Union, Callable

# Imports for specific exporters
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False
    logging.getLogger(__name__).warning("NetworkX library not found. Graph export functionalities (GEXF, GraphML) will be disabled.")

logger = logging.getLogger(__name__)

# --- GNN Parser and Helpers (from original format_exporters.py) ---

def _ensure_path(path_str: str) -> Path:
    return Path(path_str)

def _strip_comments_from_multiline_str(m_str: str) -> str:
    lines = [line.split('#', 1)[0].rstrip() for line in m_str.splitlines()]
    return "\n".join(lines).strip()

def _parse_matrix_string(matrix_str: str) -> Any:
    """Safely parses a string representation of a matrix after stripping comments."""
    
    processed_str = _strip_comments_from_multiline_str(matrix_str)
    # After stripping comments, processed_str might be empty or just whitespace
    if not processed_str:
        logger.debug(f"Matrix string was empty after comment stripping (original: '{matrix_str}')")
        return matrix_str # Or perhaps None, or an empty list, depending on desired behavior

    # Heuristic to convert GNN's array-like curly braces into valid Python list syntax for ast.literal_eval.
    # This should happen AFTER comment stripping.
    
    # New, more robust heuristic: If the string contains curly braces but no colons,
    # assume it's a nested array structure (like a multi-dimensional matrix) and convert all braces.
    # This is safer than simple replacement as it avoids accidentally converting dictionaries.
    if '{' in processed_str and ':' not in processed_str:
        processed_str = processed_str.replace('{', '[').replace('}', ']')
    # Fallback to original heuristic for single-level, non-nested structures that might be dicts.
    elif processed_str.startswith("{") and processed_str.endswith("}"):
        inner_content = processed_str[1:-1].strip()
        # If it looks like a dict (contains a colon not inside a tuple/list), leave it.
        # This is a simple check; complex dicts with nested structures might not be covered.
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
                try: return sorted(list(item))
                except TypeError: return list(item)
            elif isinstance(item, list): return [convert_structure(x) for x in item]
            elif isinstance(item, tuple): return tuple(convert_structure(x) for x in item)
            elif isinstance(item, dict): return {k: convert_structure(v) for k, v in item.items()}
            return item
        parsed_value = convert_structure(parsed_value)

        # and only contains one element that is a list/tuple, unwrap it.
        if isinstance(parsed_value, list) and len(parsed_value) == 1 and isinstance(parsed_value[0], (list,tuple)) and processed_str.startswith('[(') and processed_str.endswith(')]'):
            if processed_str.count('(') == 1 and processed_str.count(')') == 1 : # Check if it was a single tuple in original like {(...)}
                 parsed_value = list(parsed_value[0]) # Convert the inner tuple to list

        logger.debug(f"Parsed matrix string (original: '{matrix_str}') to (processed for eval: '{processed_str}'): {parsed_value}")
        return parsed_value
    except Exception as e:
        logger.warning(f"Could not parse matrix string with ast.literal_eval: '{processed_str}'. Error: {e}. Returning as raw string.")
        return matrix_str

def _parse_free_text_section(section_content: str) -> str:
    return section_content.strip()

def _parse_key_value_section(section_content: str) -> dict:
    data = {}
    for line in section_content.strip().split('\n'):
        if not line or line.startswith('#'): continue
        if '=' in line: key, value = line.split('=', 1)
        elif ':' in line: key, value = line.split(':', 1)
        else: continue
        data[key.strip()] = value.strip()
    return data

def _parse_state_line(line: str) -> Optional[dict]:
    match = re.match(r"^\s*([a-zA-Z0-9_']+)\s*(?:\[(.*?)\])?\s*(.*)$", line.split('#')[0].strip())
    if not match: return None
    state_id_default, dimensions, attributes_str = match.groups()
    attributes = {}
    if dimensions: attributes['dimensions'] = dimensions
    for kv_match in re.finditer(r'([a-zA-Z0-9_]+)\s*=\s*"([^"]*)"', attributes_str):
        attributes[kv_match.group(1)] = kv_match.group(2)
    state_id = attributes.pop('id_override', state_id_default)
    attributes['original_id'] = state_id_default
    return {"id": state_id, **attributes}

def _parse_transition_line(line: str) -> Optional[dict]:
    pattern = r"^\s*(.*?)\s*([-><]+|-)\s*(.*?)\s*(?::\s*(.*))?$"
    match = re.match(pattern, line.split('#')[0].strip())
    if not match: return None
    source_str, operator, target_str, attrs_str = match.groups()
    def clean_variable_list_str(s: str) -> List[str]:
        s = s.strip()
        if s.startswith('(') and s.endswith(')'): s = s[1:-1]
        return [v.strip() for v in s.split(',') if v.strip()]
    sources, targets = clean_variable_list_str(source_str), clean_variable_list_str(target_str)
    attributes = {}
    if attrs_str:
        for key_attr, value_attr in re.findall(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\S+)', attrs_str):
            try: attributes[key_attr] = ast.literal_eval(value_attr)
            except: attributes[key_attr] = value_attr
    return {"sources": sources, "operator": operator, "targets": targets, "attributes": attributes}

def _parse_list_items_section(content: str, parser: Callable) -> list:
    items = []
    for line in content.strip().split('\n'):
        if not line or line.startswith('#'): continue
        parsed = parser(line)
        if parsed: items.append(parsed)
    return items

def _parse_initial_parameterization_section(section_content: str) -> dict:
    data = {}
    current_key: Optional[str] = None
    current_value_lines: List[str] = []
    for line in section_content.split('\n'):
        match = re.match(r"^([a-zA-Z0-9_]+)\s*=\s*(.*)", line.strip())
        if match:
            if current_key: data[current_key] = _parse_matrix_string("\n".join(current_value_lines))
            current_key = match.group(1)
            current_value_lines = [match.group(2)]
        elif current_key:
            current_value_lines.append(line)
    if current_key: data[current_key] = _parse_matrix_string("\n".join(current_value_lines))
    return data

SECTION_PARSERS = {
    "StateSpaceBlock": lambda c: _parse_list_items_section(c, _parse_state_line),
    "Connections": lambda c: _parse_list_items_section(c, _parse_transition_line),
    "InitialParameterization": _parse_initial_parameterization_section,
    # Add other simple parsers here if needed
}

def _gnn_model_to_dict(gnn_file_path_str: str) -> dict:
    gnn_file_path = Path(gnn_file_path_str)
    if not gnn_file_path.is_file(): raise FileNotFoundError(f"GNN file not found: {gnn_file_path_str}")
    content = gnn_file_path.read_text(encoding='utf-8')
    model = {"file_path": str(gnn_file_path), "name": gnn_file_path.stem, "raw_sections": {}}
    section_regex = r"^##\s*([A-Za-z0-9_\s]+?)\s*$(.*?)(?=^##\s*[A-Za-z0-9_\s]+?\s*$|\Z)"
    for match in re.finditer(section_regex, content, re.MULTILINE | re.DOTALL):
        name, raw_content = match.group(1).strip(), match.group(2).strip()
        model["raw_sections"][name] = raw_content
        parser = SECTION_PARSERS.get(name)
        if parser: model[name.lower()] = parser(raw_content)
        elif name == "ModelName": model['name'] = raw_content
    return model

# --- XML Helpers ---
def _pretty_print_xml(element: ET.Element) -> str:
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def _dict_to_xml(tag: str, d: Union[Dict, List, Any]) -> ET.Element:
    elem = ET.Element(tag)
    if isinstance(d, dict):
        for key, val in d.items():
            safe_key = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(key))
            if not safe_key or not safe_key[0].isalpha() and safe_key[0] != '_': safe_key = '_' + safe_key
            elem.append(_dict_to_xml(safe_key, val))
    elif isinstance(d, list):
        for i, item in enumerate(d):
            elem.append(_dict_to_xml(f"{tag}_item", item))
    else:
        elem.text = str(d)
    return elem

# --- Consolidated Export Functions ---

def export_to_json_gnn(gnn_model: dict, output_file_path: str):
    """Exports the GNN model dictionary to a JSON file."""
    logger.info(f"Exporting GNN model to JSON: {output_file_path}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(gnn_model, f, indent=4, ensure_ascii=False)
        logger.debug(f"Successfully exported GNN model to JSON: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to JSON: {e}", exc_info=True)
        raise

def export_to_xml_gnn(gnn_model: dict, output_file_path: str):
    """Exports the GNN model dictionary to an XML file."""
    logger.info(f"Exporting GNN model to XML: {output_file_path}")
    try:
        root_tag_name = gnn_model.get("name", "gnn_model").replace(" ", "_")
        safe_root_tag = re.sub(r'[^a-zA-Z0-9_.-]', '_', root_tag_name)
        if not safe_root_tag or not safe_root_tag[0].isalpha() and safe_root_tag[0] != '_':
            safe_root_tag = '_' + safe_root_tag
        root_element = _dict_to_xml(safe_root_tag, gnn_model)
        xml_string = _pretty_print_xml(root_element)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        logger.debug(f"Successfully exported GNN model to XML: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to XML: {e}", exc_info=True)
        raise

def export_to_python_pickle(gnn_model: dict, output_file_path: str):
    """Serializes the GNN model dictionary to a Python pickle file."""
    logger.info(f"Exporting GNN model to Pickle: {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(gnn_model, f)
        logger.debug(f"Successfully exported GNN model to Pickle: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to Pickle: {e}", exc_info=True)
        raise

def _build_networkx_graph(gnn_model: dict) -> 'Optional[nx.DiGraph]':
    if not HAS_NETWORKX: return None
    graph = nx.DiGraph(name=gnn_model.get('name', 'GNN_Model'))
    
    # Add nodes from StateSpaceBlock (contains all variables including states, observations, etc.)
    statespace_data = gnn_model.get('statespaceblock', [])
    for node_data in statespace_data:
        if 'id' in node_data:
            # Extract node attributes
            attributes = {k: v for k, v in node_data.items() if k != 'id'}
            graph.add_node(node_data['id'], **attributes)
    
    # Add edges from Connections
    connections_data = gnn_model.get('connections', [])
    for edge_data in connections_data:
        sources = edge_data.get('sources', [])
        targets = edge_data.get('targets', [])
        operator = edge_data.get('operator', '-')
        attributes = edge_data.get('attributes', {})
        
        # Add operator information to edge attributes
        edge_attrs = attributes.copy()
        edge_attrs['operator'] = operator
        
        # Create edges from all sources to all targets
        for source in sources:
            for target in targets:
                # Ensure nodes exist (add them if they don't)
                if not graph.has_node(source):
                    graph.add_node(source, label=source)
                if not graph.has_node(target):
                    graph.add_node(target, label=target)
                
                # Add the edge
                graph.add_edge(source, target, **edge_attrs)
    
    return graph

def export_to_gexf(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a GEXF file."""
    logger.info(f"Exporting GNN model to GEXF: {output_file_path}")
    if not HAS_NETWORKX: raise ImportError("NetworkX not available, GEXF export failed.")
    graph = _build_networkx_graph(gnn_model)
    if graph is None: return
    try:
        nx.write_gexf(graph, output_file_path)
        logger.debug(f"Successfully exported to GEXF: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to GEXF: {e}", exc_info=True)
        raise

def export_to_graphml(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a GraphML file."""
    logger.info(f"Exporting GNN model to GraphML: {output_file_path}")
    if not HAS_NETWORKX: raise ImportError("NetworkX not available, GraphML export failed.")
    graph = _build_networkx_graph(gnn_model)
    if graph is None: return
    try:
        nx.write_graphml(graph, output_file_path)
        logger.debug(f"Successfully exported to GraphML: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to GraphML: {e}", exc_info=True)
        raise

def export_to_json_adjacency_list(gnn_model: dict, output_file_path: str):
    """Exports the GNN model graph to a JSON adjacency list format."""
    logger.info(f"Exporting GNN model to JSON Adjacency List: {output_file_path}")
    if not HAS_NETWORKX: raise ImportError("NetworkX not available, JSON adjacency export failed.")
    graph = _build_networkx_graph(gnn_model)
    if graph is None: return
    try:
        adj_data = nx.readwrite.json_graph.adjacency_data(graph)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(adj_data, f, indent=4)
        logger.debug(f"Successfully exported to JSON Adjacency List: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to JSON Adjacency List: {e}", exc_info=True)
        raise

def export_to_plaintext_summary(gnn_model: dict, output_file_path: str):
    """Exports a human-readable plain text summary of the GNN model."""
    logger.info(f"Exporting GNN model to Plaintext Summary: {output_file_path}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"GNN Model Summary: {gnn_model.get('name', 'N/A')}\n\n")
            for section, content in gnn_model.items():
                if section not in ['name', 'file_path', 'raw_sections'] and content:
                    f.write(f"--- {section.upper()} ---\n")
                    if isinstance(content, list):
                        for item in content: f.write(f"- {item}\n")
                    elif isinstance(content, dict):
                        for k, v in content.items(): f.write(f"- {k}: {v}\n")
                    else:
                        f.write(f"{content}\n")
                    f.write("\n")
        logger.debug(f"Successfully exported to Plaintext Summary: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to Plaintext Summary: {e}", exc_info=True)
        raise

def export_to_plaintext_dsl(gnn_model: dict, output_file_path: str):
    """Exports the GNN model back to a DSL-like format using the raw sections."""
    logger.info(f"Exporting GNN model to Plaintext DSL: {output_file_path}")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            raw_sections = gnn_model.get('raw_sections', {})
            for section_name, section_content in raw_sections.items():
                f.write(f"## {section_name}\n{section_content}\n\n")
        logger.debug(f"Successfully exported to Plaintext DSL: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export to Plaintext DSL: {e}", exc_info=True)
        raise

# This file is now the single source of truth for GNN parsing and exporting.
# The main 5_export.py script should now correctly find and use these functions. 