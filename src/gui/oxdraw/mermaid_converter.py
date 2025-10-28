"""
GNN to Mermaid Converter

Converts parsed GNN Active Inference models to Mermaid flowchart format
compatible with oxdraw editor, with embedded metadata for bidirectional sync.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re

from gnn.processor import parse_gnn_file


def gnn_to_mermaid(
    gnn_model: Dict[str, Any],
    include_metadata: bool = True,
    include_styling: bool = True
) -> str:
    """
    Convert parsed GNN model to Mermaid flowchart format.
    
    Args:
        gnn_model: Parsed GNN model dictionary
        include_metadata: Include oxdraw-compatible metadata in comments
        include_styling: Include node/edge styling directives
        
    Returns:
        Mermaid flowchart string with embedded GNN metadata
    """
    lines = []
    
    # Header with flowchart directive
    lines.append("flowchart TD")
    
    # Model metadata in comments
    model_name = gnn_model.get('model_name', 'Untitled Model')
    version = gnn_model.get('version', '1.0')
    
    lines.append(f"    %% GNN Model: {model_name}")
    lines.append(f"    %% GNN Version: {version}")
    lines.append("")
    
    # Embed full GNN specification as JSON in comment
    if include_metadata:
        metadata = generate_mermaid_metadata(gnn_model)
        # Split long metadata across multiple comment lines for readability
        metadata_json = json.dumps(metadata, separators=(',', ':'))
        lines.append(f"    %% GNN_METADATA_START")
        lines.append(f"    %% {metadata_json}")
        lines.append(f"    %% GNN_METADATA_END")
        lines.append("")
    
    # Generate nodes from variables
    variables = gnn_model.get('variables', {})
    
    if variables:
        lines.append("    %% === Nodes (Variables) ===")
        
        # Handle dict, list of dicts, or list of strings formats
        if isinstance(variables, dict):
            for var_name, var_data in variables.items():
                node_def = _generate_node_definition(var_name, var_data)
                lines.append(f"    {node_def}")
                
                # Add ontology annotation as comment
                ontology = var_data.get('ontology_mapping', '')
                if ontology:
                    lines.append(f"    %% {var_name}: {ontology}")
        elif isinstance(variables, list):
            for var in variables:
                if isinstance(var, dict):
                    # List of dicts
                    var_name = var.get('name', '')
                    if var_name:
                        node_def = _generate_node_definition(var_name, var)
                        lines.append(f"    {node_def}")
                        
                        # Add ontology annotation as comment
                        ontology = var.get('ontology_mapping', '')
                        if ontology:
                            lines.append(f"    %% {var_name}: {ontology}")
                elif isinstance(var, str):
                    # List of strings (lightweight parser) - use basic node
                    lines.append(f"    {var}[{var}]")
        
        lines.append("")
    
    # Generate edges from connections
    connections = gnn_model.get('connections', [])
    
    if connections:
        lines.append("    %% === Edges (Connections) ===")
        
        for conn in connections:
            edge_def = _generate_edge_definition(conn)
            lines.append(f"    {edge_def}")
            
            # Add connection type as comment
            conn_type = conn.get('connection_type', '')
            if conn_type:
                lines.append(f"    %% Connection: {conn_type}")
        
        lines.append("")
    
    # Styling section
    if include_styling and variables:
        lines.append("    %% === Styling ===")
        style_defs = _generate_node_styles(variables)
        lines.extend([f"    {style}" for style in style_defs])
    
    return "\n".join(lines)


def _generate_node_definition(var_name: str, var_data: Dict[str, Any]) -> str:
    """
    Generate Mermaid node definition from GNN variable.
    
    Returns string like: A[A<br/>3x3<br/>float]
    """
    from .utils import infer_node_shape
    
    # Get shape based on variable characteristics
    shape_open, shape_close = infer_node_shape(var_name, var_data)
    
    # Generate label with dimensions and type
    label_parts = [var_name]
    
    # Handle both 'dimensions' and nested structure
    dimensions = var_data.get('dimensions', [])
    if dimensions:
        dims_str = 'x'.join(str(d) for d in dimensions)
        label_parts.append(dims_str)
    
    # Handle both 'data_type' and 'type'
    data_type = var_data.get('data_type') or var_data.get('type', '')
    if data_type:
        label_parts.append(data_type)
    
    label = '<br/>'.join(label_parts)
    
    return f"{var_name}{shape_open}{label}{shape_close}"


def _generate_edge_definition(conn: Dict[str, Any]) -> str:
    """
    Generate Mermaid edge definition from GNN connection.
    
    Returns string like: D ==> s
    """
    from .utils import infer_edge_style
    
    source = conn.get('source', '')
    target = conn.get('target', '')
    symbol = conn.get('symbol', '')
    description = conn.get('description', '')
    
    # Get edge style based on connection symbol
    edge_style = infer_edge_style(symbol)
    
    # Add label if description exists
    if description:
        # Sanitize description for Mermaid
        safe_desc = description.replace('|', '\\|')
        edge_def = f"{source} {edge_style}|{safe_desc}| {target}"
    else:
        edge_def = f"{source} {edge_style} {target}"
    
    return edge_def


def _generate_node_styles(variables: Dict[str, Any]) -> List[str]:
    """
    Generate Mermaid styling directives based on variable types.
    
    Returns list of style definition strings.
    """
    styles = []
    
    # Define style classes
    style_classes = {
        'matrix': 'fill:#e1f5ff,stroke:#0288d1,stroke-width:2px',
        'vector': 'fill:#fff9e6,stroke:#fbc02d,stroke-width:2px',
        'state': 'fill:#fff3e0,stroke:#f57c00,stroke-width:2px',
        'observation': 'fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px',
        'action': 'fill:#e8f5e9,stroke:#388e3c,stroke-width:2px',
        'policy': 'fill:#e3f2fd,stroke:#1976d2,stroke-width:2px',
        'free_energy': 'fill:#fce4ec,stroke:#c2185b,stroke-width:2px'
    }
    
    # Group variables by type
    var_groups: Dict[str, List[str]] = {
        'matrix': [],
        'vector': [],
        'state': [],
        'observation': [],
        'action': [],
        'policy': [],
        'free_energy': []
    }
    
    # Handle both dict and list formats
    if isinstance(variables, dict):
        for var_name, var_data in variables.items():
            var_type = _classify_variable(var_name, var_data)
            if var_type in var_groups:
                var_groups[var_type].append(var_name)
    elif isinstance(variables, list):
        for var in variables:
            if isinstance(var, dict):
                var_name = var.get('name', '')
                if var_name:
                    var_type = _classify_variable(var_name, var)
                    if var_type in var_groups:
                        var_groups[var_type].append(var_name)
            elif isinstance(var, str):
                # Simple string names - classify as vector by default
                var_groups['vector'].append(var)
    
    # Generate style definitions
    for style_type, style_def in style_classes.items():
        if var_groups[style_type]:
            # Define class
            styles.append(f"classDef {style_type}Style {style_def}")
            # Apply to variables
            for var_name in var_groups[style_type]:
                styles.append(f"class {var_name} {style_type}Style")
    
    return styles


def _classify_variable(var_name: str, var_data: Dict[str, Any]) -> str:
    """
    Classify variable into type category for styling.
    
    Returns: 'matrix', 'vector', 'state', 'observation', 'action', 'policy', 'free_energy'
    """
    ontology = var_data.get('ontology_mapping', '')
    dims = var_data.get('dimensions', [])
    
    # Check ontology mapping first (most specific)
    if 'State' in ontology:
        return 'state'
    elif 'Observation' in ontology:
        return 'observation'
    elif 'Action' in ontology:
        return 'action'
    elif 'Policy' in ontology or var_name in ['π', 'pi']:
        return 'policy'
    elif 'FreeEnergy' in ontology or var_name in ['F', 'G']:
        return 'free_energy'
    
    # Check dimensionality second
    if len(dims) >= 2:
        return 'matrix'
    else:
        return 'vector'


def generate_mermaid_metadata(gnn_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate metadata dictionary for embedding in Mermaid comments.
    
    Args:
        gnn_model: Parsed GNN model
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "model_name": gnn_model.get('model_name', 'Untitled'),
        "version": gnn_model.get('version', '1.0'),
        "variables": {},
        "connections": [],
        "parameters": gnn_model.get('parameters', {}),
        "ontology_mappings": {}
    }
    
    # Handle variables - can be dict, list of dicts, or list of strings
    variables = gnn_model.get('variables', {})
    
    if isinstance(variables, dict):
        # Dictionary format
        for var_name, var_data in variables.items():
            metadata["variables"][var_name] = {
                "dimensions": var_data.get('dimensions', []),
                "data_type": var_data.get('data_type', 'float'),
                "ontology_mapping": var_data.get('ontology_mapping', ''),
                "description": var_data.get('description', '')
            }
            
            # Add to ontology mappings if present
            if var_data.get('ontology_mapping'):
                metadata["ontology_mappings"][var_name] = var_data['ontology_mapping']
    
    elif isinstance(variables, list):
        # List format - can be list of dicts or list of strings
        for var in variables:
            if isinstance(var, dict):
                # List of dict objects
                var_name = var.get('name', '')
                if var_name:
                    metadata["variables"][var_name] = {
                        "dimensions": var.get('dimensions', []),
                        "data_type": var.get('type', 'float'),
                        "ontology_mapping": var.get('ontology_mapping', ''),
                        "description": var.get('description', var.get('value', ''))
                    }
            elif isinstance(var, str):
                # List of strings (lightweight parser)
                metadata["variables"][var] = {
                    "dimensions": [],
                    "data_type": "float",
                    "ontology_mapping": "",
                    "description": ""
                }
    
    # Serialize connections
    for conn in gnn_model.get('connections', []):
        metadata["connections"].append({
            "source": conn.get('source', ''),
            "target": conn.get('target', ''),
            "symbol": conn.get('symbol', conn.get('type', '')),
            "connection_type": conn.get('connection_type', conn.get('type', ''))
        })
    
    return metadata


def convert_gnn_file_to_mermaid(
    gnn_file_path: Path,
    output_path: Optional[Path] = None,
    include_metadata: bool = True,
    include_styling: bool = True
) -> str:
    """
    Convert a GNN file to Mermaid format for oxdraw.
    
    Args:
        gnn_file_path: Path to GNN markdown file
        output_path: Optional path to save Mermaid output
        include_metadata: Include metadata for bidirectional sync
        include_styling: Include visual styling
        
    Returns:
        Mermaid diagram string
    """
    # Parse GNN file using existing pipeline module
    parsed_model = parse_gnn_file(gnn_file_path)
    
    # Convert to Mermaid
    mermaid_content = gnn_to_mermaid(
        parsed_model,
        include_metadata=include_metadata,
        include_styling=include_styling
    )
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(mermaid_content, encoding='utf-8')
    
    return mermaid_content

