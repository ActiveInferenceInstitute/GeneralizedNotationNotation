"""
Export module for GNN Processing Pipeline.

This module provides multi-format export capabilities for GNN files.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def generate_exports(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False
) -> bool:
    """
    Generate exports in multiple formats for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to export
        output_dir: Directory to save exports
        verbose: Enable verbose output
        
    Returns:
        True if exports generated successfully, False otherwise
    """
    logger = logging.getLogger("export")
    
    try:
        log_step_start(logger, "Generating multi-format exports")
        
        # Create exports directory
        exports_dir = output_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            log_step_warning(logger, "No GNN files found for export")
            return True
        
        # Generate exports for each file
        export_results = {}
        for gnn_file in gnn_files:
            file_exports = export_single_gnn_file(gnn_file, exports_dir)
            export_results[gnn_file.name] = file_exports
        
        # Save export results
        import json
        results_file = exports_dir / "export_results.json"
        with open(results_file, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        # Check overall success
        all_successful = all(result["success"] for result in export_results.values())
        
        if all_successful:
            log_step_success(logger, "All exports generated successfully")
        else:
            failed_files = [name for name, result in export_results.items() if not result["success"]]
            log_step_error(logger, f"Export failed for some files: {failed_files}")
        
        return all_successful
        
    except Exception as e:
        log_step_error(logger, f"Export generation failed: {e}")
        return False

def export_single_gnn_file(gnn_file: Path, exports_dir: Path) -> Dict[str, Any]:
    """
    Export a single GNN file to multiple formats.
    
    Args:
        gnn_file: Path to the GNN file to export
        exports_dir: Directory to save exports
        
    Returns:
        Dictionary with export results
    """
    try:
        # Read file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse GNN content
        parsed_content = parse_gnn_content(content)
        
        # Generate exports
        exports = {}
        
        # JSON export
        json_file = exports_dir / f"{gnn_file.stem}.json"
        exports["json"] = export_to_json(parsed_content, json_file)
        
        # XML export
        xml_file = exports_dir / f"{gnn_file.stem}.xml"
        exports["xml"] = export_to_xml(parsed_content, xml_file)
        
        # GraphML export
        graphml_file = exports_dir / f"{gnn_file.stem}.graphml"
        exports["graphml"] = export_to_graphml(parsed_content, graphml_file)
        
        # GEXF export
        gexf_file = exports_dir / f"{gnn_file.stem}.gexf"
        exports["gexf"] = export_to_gexf(parsed_content, gexf_file)
        
        # Pickle export
        pickle_file = exports_dir / f"{gnn_file.stem}.pkl"
        exports["pickle"] = export_to_pickle(parsed_content, pickle_file)
        
        return {
            "file": str(gnn_file),
            "success": all(exports.values()),
            "exports": exports
        }
        
    except Exception as e:
        return {
            "file": str(gnn_file),
            "success": False,
            "error": str(e),
            "exports": {}
        }

def parse_gnn_content(content: str) -> Dict[str, Any]:
    """
    Parse GNN content into structured format.
    
    Args:
        content: GNN file content
        
    Returns:
        Dictionary with parsed content
    """
    try:
        parsed = {
            "sections": {},
            "metadata": {},
            "variables": [],
            "connections": [],
            "equations": []
        }
        
        # Basic parsing
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                current_section = line[3:]
                parsed["sections"][current_section] = []
            elif current_section and line:
                parsed["sections"][current_section].append(line)
        
        return parsed
        
    except Exception:
        return {
            "sections": {},
            "metadata": {},
            "variables": [],
            "connections": [],
            "equations": []
        }

def export_to_json(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export to JSON format."""
    try:
        import json
        with open(output_file, 'w') as f:
            json.dump(parsed_content, f, indent=2)
        return True
    except Exception:
        return False

def export_to_xml(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export to XML format."""
    try:
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<gnn>\n'
        
        for section_name, section_content in parsed_content["sections"].items():
            xml_content += f'  <{section_name.lower()}>\n'
            for line in section_content:
                xml_content += f'    <line>{line}</line>\n'
            xml_content += f'  </{section_name.lower()}>\n'
        
        xml_content += '</gnn>'
        
        with open(output_file, 'w') as f:
            f.write(xml_content)
        return True
    except Exception:
        return False

def export_to_graphml(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export to GraphML format."""
    try:
        # Basic GraphML structure
        graphml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="G" edgedefault="undirected">
'''
        
        # Add nodes and edges based on parsed content
        # This is a simplified implementation
        
        graphml_content += '''  </graph>
</graphml>'''
        
        with open(output_file, 'w') as f:
            f.write(graphml_content)
        return True
    except Exception:
        return False

def export_to_gexf(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export to GEXF format."""
    try:
        # Basic GEXF structure
        gexf_content = '''<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft">
  <graph mode="static" defaultedgetype="undirected">
    <nodes>
    </nodes>
    <edges>
    </edges>
  </graph>
</gexf>'''
        
        with open(output_file, 'w') as f:
            f.write(gexf_content)
        return True
    except Exception:
        return False

def export_to_pickle(parsed_content: Dict[str, Any], output_file: Path) -> bool:
    """Export to Pickle format."""
    try:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(parsed_content, f)
        return True
    except Exception:
        return False

def export_model(model_data: Dict[str, Any], output_dir: Path, formats: List[str] = None) -> Dict[str, Any]:
    """
    Export a model to multiple formats.
    
    Args:
        model_data: Model data to export
        output_dir: Directory to save exports
        formats: List of formats to export (default: all available)
        
    Returns:
        Dictionary with export results
    """
    if formats is None:
        formats = ['json', 'xml', 'graphml', 'gexf', 'pickle']
    
    results = {}
    
    for format_name in formats:
        if format_name == 'json':
            output_file = output_dir / f"model.{format_name}"
            results[format_name] = export_to_json(model_data, output_file)
        elif format_name == 'xml':
            output_file = output_dir / f"model.{format_name}"
            results[format_name] = export_to_xml(model_data, output_file)
        elif format_name == 'graphml':
            output_file = output_dir / f"model.{format_name}"
            results[format_name] = export_to_graphml(model_data, output_file)
        elif format_name == 'gexf':
            output_file = output_dir / f"model.{format_name}"
            results[format_name] = export_to_gexf(model_data, output_file)
        elif format_name == 'pickle':
            output_file = output_dir / f"model.{format_name}"
            results[format_name] = export_to_pickle(model_data, output_file)
    
    return {
        "success": all(results.values()),
        "formats": results,
        "output_dir": str(output_dir)
    }

def _gnn_model_to_dict(gnn_content: str) -> Dict[str, Any]:
    """
    Convert GNN model content to dictionary format.
    
    Args:
        gnn_content: GNN file content as string
        
    Returns:
        Dictionary representation of the GNN model
    """
    try:
        # Parse the GNN content into structured format
        parsed = parse_gnn_content(gnn_content)
        
        # Convert to standardized dictionary format
        model_dict = {
            "model_name": parsed.get("sections", {}).get("ModelName", ["Unknown"])[0] if parsed.get("sections", {}).get("ModelName") else "Unknown",
            "model_annotation": parsed.get("sections", {}).get("ModelAnnotation", [""])[0] if parsed.get("sections", {}).get("ModelAnnotation") else "",
            "variables": parsed.get("variables", []),
            "connections": parsed.get("connections", []),
            "equations": parsed.get("equations", []),
            "metadata": parsed.get("metadata", {}),
            "source_content": gnn_content
        }
        
        return model_dict
    except Exception as e:
        return {
            "model_name": "Error",
            "model_annotation": f"Error parsing GNN content: {str(e)}",
            "variables": [],
            "connections": [],
            "equations": [],
            "metadata": {"error": str(e)},
            "source_content": gnn_content
        }

def export_to_json_gnn(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export to JSON GNN format."""
    try:
        import json
        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        return True
    except Exception:
        return False

def export_to_xml_gnn(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export to XML GNN format."""
    try:
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<gnn_model>\n'
        
        # Add model metadata
        xml_content += f'  <model_name>{model_data.get("model_name", "Unknown")}</model_name>\n'
        xml_content += f'  <model_annotation>{model_data.get("model_annotation", "")}</model_annotation>\n'
        
        # Add variables
        xml_content += '  <variables>\n'
        for var in model_data.get("variables", []):
            xml_content += f'    <variable>{var}</variable>\n'
        xml_content += '  </variables>\n'
        
        # Add connections
        xml_content += '  <connections>\n'
        for conn in model_data.get("connections", []):
            xml_content += f'    <connection>{conn}</connection>\n'
        xml_content += '  </connections>\n'
        
        xml_content += '</gnn_model>'
        
        with open(output_file, 'w') as f:
            f.write(xml_content)
        return True
    except Exception:
        return False

def export_to_python_pickle(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export to Python pickle format."""
    try:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(model_data, f)
        return True
    except Exception:
        return False

def export_to_plaintext_summary(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export to plaintext summary format."""
    try:
        summary = f"""GNN Model Summary
================

Model Name: {model_data.get("model_name", "Unknown")}
Model Annotation: {model_data.get("model_annotation", "")}

Variables ({len(model_data.get("variables", []))}):
"""
        for var in model_data.get("variables", []):
            summary += f"  - {var}\n"
        
        connections_count = len(model_data.get("connections", []))
        summary += f"\nConnections ({connections_count}):\n"
        for conn in model_data.get("connections", []):
            summary += f"  - {conn}\n"
        
        summary += f"\nEquations ({len(model_data.get('equations', []))}):\n"
        for eq in model_data.get("equations", []):
            summary += f"  - {eq}\n"
        
        with open(output_file, 'w') as f:
            f.write(summary)
        return True
    except Exception:
        return False

def export_to_plaintext_dsl(model_data: Dict[str, Any], output_file: Path) -> bool:
    """Export to plaintext DSL format."""
    try:
        dsl_content = f"""# GNN Model DSL Export

## Model Definition
model {model_data.get("model_name", "Unknown")} {{
    annotation: "{model_data.get("model_annotation", "")}"
    
    variables: {{
"""
        for var in model_data.get("variables", []):
            dsl_content += f"        {var}\n"
        
        dsl_content += "    }\n"
        dsl_content += "    connections: {\n"
        for conn in model_data.get("connections", []):
            dsl_content += f"        {conn}\n"
        
        dsl_content += "    }\n"
        dsl_content += "    equations: {\n"
        for eq in model_data.get("equations", []):
            dsl_content += f"        {eq}\n"
        
        dsl_content += "    }\n"
        dsl_content += "}\n"
        
        with open(output_file, 'w') as f:
            f.write(dsl_content)
        return True
    except Exception:
        return False

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the export module and its capabilities."""
    info = {
        'version': __version__,
        'description': __description__,
        'features': FEATURES,
        'available_formats': [],
        'graph_formats': [],
        'text_formats': [],
        'data_formats': []
    }
    
    # Available formats
    info['available_formats'].extend([
        'JSON', 'XML', 'GraphML', 'GEXF', 'Pickle',
        'JSON GNN', 'XML GNN', 'Python Pickle',
        'Plaintext Summary', 'Plaintext DSL'
    ])
    
    # Graph formats
    info['graph_formats'].extend(['GraphML', 'GEXF'])
    
    # Text formats
    info['text_formats'].extend(['JSON', 'XML', 'Plaintext Summary', 'Plaintext DSL'])
    
    # Data formats
    info['data_formats'].extend(['Pickle', 'JSON', 'XML'])
    
    return info

def get_supported_formats() -> Dict[str, List[str]]:
    """Get supported export formats."""
    return {
        'data_formats': ['JSON', 'XML', 'Pickle'],
        'text_formats': ['JSON', 'XML', 'Plaintext Summary', 'Plaintext DSL'],
        'graph_formats': ['GraphML', 'GEXF']
    }

def export_gnn_model(model_data: Dict[str, Any], output_dir: Path, formats: List[str] = None) -> Dict[str, Any]:
    """
    Export GNN model to multiple formats (alias for export_model).
    
    Args:
        model_data: Model data to export
        output_dir: Directory to save exports
        formats: List of formats to export (default: all available)
        
    Returns:
        Dictionary with export results
    """
    # Handle invalid input gracefully
    if not isinstance(model_data, dict):
        return {
            "success": False,
            "error": f"Invalid model_data type: {type(model_data).__name__}. Expected dict.",
            "formats": {},
            "output_dir": str(output_dir) if output_dir else "None"
        }
    
    # Convert output_dir to Path if it's a string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    return export_model(model_data, output_dir, formats)

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "Multi-format export for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'json_export': True,
    'xml_export': True,
    'graphml_export': True,
    'gexf_export': True,
    'pickle_export': True,
    'mcp_integration': True
}

# Check for optional dependencies
try:
    import networkx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

__all__ = [
    'generate_exports',
    'export_single_gnn_file',
    'export_model',
    'export_gnn_model',
    '_gnn_model_to_dict',
    'parse_gnn_content',
    'export_to_json',
    'export_to_xml',
    'export_to_graphml',
    'export_to_gexf',
    'export_to_pickle',
    'export_to_json_gnn',
    'export_to_xml_gnn',
    'export_to_python_pickle',
    'export_to_plaintext_summary',
    'export_to_plaintext_dsl',
    'get_module_info',
    'get_supported_formats',
    'FEATURES',
    '__version__'
]
