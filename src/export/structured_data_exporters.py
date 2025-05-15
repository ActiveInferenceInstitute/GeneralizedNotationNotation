"""
Specialized GNN Exporters for Structured Data Formats (JSON, XML, Python Pickle)
"""
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle
import logging
import re # For _dict_to_xml key sanitization
from typing import Dict, Any, List, Optional, Union # For _dict_to_xml type hints

logger = logging.getLogger(__name__)

# --- XML Helper (copied from original format_exporters.py) ---
def _pretty_print_xml(element: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def _dict_to_xml(tag: str, d: Union[Dict[str, Any], List[Any], str, int, float, bool, None]) -> ET.Element:
    """Recursively convert a Python dictionary or list to an XML ET.Element."""
    elem = ET.Element(tag)
    if isinstance(d, dict):
        for key, val in d.items():
            safe_key = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(key)) # Allow . and - in tags if they are not first char
            if not safe_key or not safe_key[0].isalpha() and safe_key[0] != '_':
                safe_key = '_' + safe_key
            # Replace special characters that are still problematic even if not first
            safe_key = safe_key.replace("[", "_ob_").replace("]", "_cb_").replace("(", "_op_").replace(")", "_cp_").replace("{", "_ocb_").replace("}", "_ccb_")
            
            child = _dict_to_xml(safe_key, val)
            elem.append(child)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            item_tag = f"{tag}_item" 
            child = _dict_to_xml(item_tag, item)
            elem.append(child)
    elif d is None:
        elem.text = "" 
    else:
        elem.text = str(d)
    return elem

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
        # Ensure the top-level GNN model dictionary has a root tag name if being converted directly
        # For example, if gnn_model itself is the d in _dict_to_xml(tag, d)
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
        logger.error(f"Failed to export GNN model to XML {output_file_path}: {e}", exc_info=True)
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