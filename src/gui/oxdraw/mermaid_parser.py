"""
Mermaid to GNN Parser

Extracts GNN model structure from Mermaid flowcharts with embedded metadata,
enabling bidirectional synchronization with oxdraw editor.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re


def mermaid_to_gnn(
    mermaid_content: str,
    validate_ontology: bool = False
) -> Dict[str, Any]:
    """
    Parse Mermaid flowchart back to GNN model structure.
    
    Args:
        mermaid_content: Mermaid flowchart string from oxdraw
        validate_ontology: Validate ontology term mappings (requires ontology module)
        
    Returns:
        GNN model dictionary ready for pipeline processing
    """
    # Extract metadata from comments
    metadata = extract_gnn_metadata(mermaid_content)
    
    # Parse visual structure (nodes and edges)
    nodes = _extract_nodes(mermaid_content)
    edges = _extract_edges(mermaid_content)
    
    # Merge metadata with visual edits
    variables = _merge_variables(metadata.get('variables', {}), nodes)
    connections = _merge_connections(metadata.get('connections', []), edges)
    
    # Validate ontology mappings if requested
    if validate_ontology and metadata.get('ontology_mappings'):
        try:
            from ontology.processor import load_defined_ontology_terms, validate_annotations
            
            ontology_terms = load_defined_ontology_terms()
            validation_result = validate_annotations(
                list(metadata['ontology_mappings'].values()),
                ontology_terms
            )
            
            if validation_result.get('invalid_annotations'):
                raise ValueError(
                    f"Invalid ontology terms: {validation_result['invalid_annotations']}"
                )
        except ImportError:
            # Ontology module not available, skip validation
            pass
    
    # Construct GNN model dictionary
    gnn_model = {
        "model_name": metadata.get('model_name', 'Untitled Model'),
        "version": metadata.get('version', '1.0'),
        "variables": variables,
        "connections": connections,
        "parameters": metadata.get('parameters', {}),
        "ontology_mappings": _reconstruct_ontology_mappings(
            variables,
            metadata.get('ontology_mappings', {})
        )
    }
    
    return gnn_model


def extract_gnn_metadata(mermaid_content: str) -> Dict[str, Any]:
    """
    Extract embedded GNN metadata from Mermaid comments.
    
    Looks for metadata between GNN_METADATA_START and GNN_METADATA_END markers.
    
    Args:
        mermaid_content: Mermaid diagram content
        
    Returns:
        Metadata dictionary
    """
    # Try new multi-line format first
    metadata_pattern = r'%%\s*GNN_METADATA_START.*?%%\s*(.+?)%%\s*GNN_METADATA_END'
    match = re.search(metadata_pattern, mermaid_content, re.DOTALL)
    
    if match:
        try:
            metadata_json = match.group(1).strip()
            return json.loads(metadata_json)
        except json.JSONDecodeError:
            pass
    
    # Try legacy single-line format
    legacy_pattern = r'%%\s*GNN_METADATA:\s*(\{.*?\})'
    match = re.search(legacy_pattern, mermaid_content, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    return {}


def _extract_nodes(mermaid_content: str) -> Dict[str, Dict[str, Any]]:
    r"""
    Extract node definitions from Mermaid diagram.
    
    Supports various Mermaid shapes:
    - [A] : Rectangle
    - (C) : Rounded
    - ([s]) : Stadium
    - ((o)) : Circle
    - {{u}} : Hexagon
    - {π} : Diamond
    - [/F\] or [\F/] : Trapezoid
    
    Returns:
        Dictionary mapping node IDs to node metadata
    """
    nodes = {}
    
    # Pattern definitions for various node shapes
    patterns = [
        (r'(\w+)\[([^\]]+)\]', 'rectangle'),            # [A] - must come before stadium
        (r'(\w+)\(\[([^\]]+)\]\)', 'stadium'),           # ([s])
        (r'(\w+)\(\(([^)]+)\)\)', 'circle'),             # ((o))
        (r'(\w+)\{\{([^}]+)\}\}', 'hexagon'),            # {{u}}
        (r'(\w+)\{([^}]+)\}', 'diamond'),                # {π}
        (r'(\w+)\[\/([^\\]+)\\\]', 'trapezoid'),         # [/F\]
        (r'(\w+)\[\\([^/]+)\/\]', 'trapezoid_inv'),     # [\F/]
        (r'(\w+)\(([^)]+)\)', 'rounded'),                # (C) - must come after stadium and circle
    ]
    
    for pattern, shape in patterns:
        for match in re.finditer(pattern, mermaid_content):
            node_id = match.group(1)
            node_label = match.group(2)
            
            # Skip if already found (first match wins)
            if node_id in nodes:
                continue
            
            # Parse label for dimensions and type
            label_parts = node_label.split('<br/>')
            
            nodes[node_id] = {
                'shape': shape,
                'label': node_label,
                'label_parts': label_parts,
                'inferred_dimensions': _infer_dimensions_from_label(label_parts),
                'inferred_type': _infer_type_from_label(label_parts)
            }
    
    return nodes


def _extract_edges(mermaid_content: str) -> List[Dict[str, Any]]:
    """
    Extract edge definitions from Mermaid diagram.
    
    Supports edge styles:
    - ==> : Thick arrow (generative)
    - -.-> : Dashed (inference)
    - -..-> : Dotted (modulation)
    - --> : Normal arrow (coupling)
    
    Returns:
        List of edge dictionaries
    """
    edges = []
    
    edge_patterns = [
        (r'(\w+)\s*==>\|([^\|]+)\|\s*(\w+)', '>', True),   # Generative with label
        (r'(\w+)\s*==>\s*(\w+)', '>', False),               # Generative
        (r'(\w+)\s*-\.\->\|([^\|]+)\|\s*(\w+)', '-', True), # Inference with label
        (r'(\w+)\s*-\.\->\s*(\w+)', '-', False),            # Inference
        (r'(\w+)\s*-\.\.\->\|([^\|]+)\|\s*(\w+)', '*', True), # Modulation with label
        (r'(\w+)\s*-\.\.\->\s*(\w+)', '*', False),          # Modulation
        (r'(\w+)\s*-->\|([^\|]+)\|\s*(\w+)', '~', True),    # Coupling with label
        (r'(\w+)\s*-->\s*(\w+)', '~', False)                # Coupling
    ]
    
    for pattern, symbol, has_label in edge_patterns:
        for match in re.finditer(pattern, mermaid_content):
            if has_label:
                source = match.group(1)
                label = match.group(2)
                target = match.group(3)
            else:
                source = match.group(1)
                target = match.group(2)
                label = ''
            
            edges.append({
                'source': source,
                'target': target,
                'symbol': symbol,
                'description': label,
                'line_position': match.start()
            })
    
    return edges


def _infer_dimensions_from_label(label_parts: List[str]) -> List[int]:
    """
    Infer variable dimensions from node label.
    
    Looks for patterns like "3x3" or "3x3x3" in label parts.
    """
    for part in label_parts:
        dim_match = re.match(r'^(\d+)(?:x(\d+))?(?:x(\d+))?$', part.strip())
        if dim_match:
            dims = [int(d) for d in dim_match.groups() if d is not None]
            return dims
    
    return []


def _infer_type_from_label(label_parts: List[str]) -> str:
    """
    Infer data type from node label.
    
    Looks for type hints like "float", "int", "categorical" in label parts.
    """
    type_keywords = ['float', 'int', 'categorical', 'bool']
    
    for part in label_parts:
        part_lower = part.strip().lower()
        if part_lower in type_keywords:
            return part_lower
    
    return 'float'  # Default


def _merge_variables(
    metadata_vars: Dict[str, Any],
    visual_nodes: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge metadata variables with visually edited node positions.
    
    Preserves metadata dimensions/types while respecting visual layout changes.
    Visual structure takes precedence for topology.
    """
    merged = {}
    
    # Start with metadata (preserves complete information)
    for var_name, var_data in metadata_vars.items():
        merged[var_name] = var_data.copy()
    
    # Update/add from visual structure
    for node_id, node_data in visual_nodes.items():
        if node_id not in merged:
            # New node added in visual editor
            merged[node_id] = {
                'dimensions': node_data['inferred_dimensions'] or [3, 1],
                'data_type': node_data['inferred_type'],
                'description': node_data['label_parts'][0] if node_data['label_parts'] else node_id
            }
        else:
            # Existing node - update description if changed
            if node_data['label_parts']:
                merged[node_id]['description'] = node_data['label_parts'][0]
    
    return merged


def _merge_connections(
    metadata_conns: List[Dict[str, Any]],
    visual_edges: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge metadata connections with visually edited edges.
    
    Visual edits (adding/removing edges in oxdraw) take precedence over metadata.
    """
    merged = []
    
    for edge in visual_edges:
        # Find matching metadata connection
        metadata_conn = next(
            (c for c in metadata_conns
             if c['source'] == edge['source'] and c['target'] == edge['target']),
            None
        )
        
        merged.append({
            'source': edge['source'],
            'target': edge['target'],
            'symbol': edge['symbol'],
            'connection_type': metadata_conn.get('connection_type', 'directed') if metadata_conn else 'directed',
            'description': edge.get('description') or (metadata_conn.get('description', '') if metadata_conn else '')
        })
    
    return merged


def _reconstruct_ontology_mappings(
    variables: Dict[str, Any],
    ontology_map: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Reconstruct ontology mapping list from merged variables.
    
    Returns list of {variable, ontology_term} mappings.
    """
    mappings = []
    
    for var_name, var_data in variables.items():
        ontology_term = var_data.get('ontology_mapping') or ontology_map.get(var_name)
        if ontology_term:
            mappings.append({
                'variable': var_name,
                'ontology_term': ontology_term
            })
    
    return mappings


def convert_mermaid_file_to_gnn(
    mermaid_file_path: Path,
    output_path: Optional[Path] = None,
    validate_ontology: bool = False
) -> Dict[str, Any]:
    """
    Convert Mermaid file from oxdraw back to GNN format.
    
    Args:
        mermaid_file_path: Path to Mermaid file (.mmd)
        output_path: Optional path to save GNN output (.md)
        validate_ontology: Validate ontology term mappings
        
    Returns:
        Parsed GNN model dictionary
    """
    mermaid_content = mermaid_file_path.read_text(encoding='utf-8')
    parsed_model = mermaid_to_gnn(mermaid_content, validate_ontology=validate_ontology)
    
    # Convert to GNN markdown if output path provided
    if output_path:
        gnn_content = _gnn_model_to_markdown(parsed_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(gnn_content, encoding='utf-8')
    
    return parsed_model


def _gnn_model_to_markdown(gnn_model: Dict[str, Any]) -> str:
    """
    Convert GNN model dictionary to markdown format.
    
    Generates a valid GNN specification file.
    """
    lines = []
    
    # Header
    lines.append(f"# GNN Model: {gnn_model.get('model_name', 'Untitled')}")
    lines.append(f"# GNN Version: {gnn_model.get('version', '1.0')}")
    lines.append(f"# Generated from Mermaid diagram via oxdraw integration")
    lines.append("")
    
    # Model name section
    lines.append("## ModelName")
    lines.append(gnn_model.get('model_name', 'Untitled Model'))
    lines.append("")
    
    # State space block
    lines.append("## StateSpaceBlock")
    for var_name, var_data in gnn_model.get('variables', {}).items():
        dims = var_data.get('dimensions', [])
        dtype = var_data.get('data_type', 'float')
        desc = var_data.get('description', '')
        
        if dims:
            dims_str = ','.join(str(d) for d in dims)
            lines.append(f"{var_name}[{dims_str},type={dtype}]  # {desc}")
        else:
            lines.append(f"{var_name}[type={dtype}]  # {desc}")
    
    lines.append("")
    
    # Connections
    lines.append("## Connections")
    for conn in gnn_model.get('connections', []):
        source = conn.get('source', '')
        target = conn.get('target', '')
        symbol = conn.get('symbol', '')
        lines.append(f"{source}{symbol}{target}")
    
    lines.append("")
    
    # Ontology annotations
    if gnn_model.get('ontology_mappings'):
        lines.append("## ActInfOntologyAnnotation")
        for mapping in gnn_model['ontology_mappings']:
            var = mapping.get('variable', '')
            term = mapping.get('ontology_term', '')
            lines.append(f"{var}={term}")
        lines.append("")
    
    # Parameters
    if gnn_model.get('parameters'):
        lines.append("## ModelParameters")
        for param_name, param_value in gnn_model['parameters'].items():
            lines.append(f"{param_name}: {param_value}")
        lines.append("")
    
    return "\n".join(lines)

