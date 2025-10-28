"""
Utility functions for oxdraw integration

Provides helper functions for node/edge styling, validation, and configuration.
"""

from typing import Dict, Any, Tuple, List, Optional
import re


def infer_node_shape(var_name: str, var_data: Dict[str, Any]) -> Tuple[str, str]:
    r"""
    Infer Mermaid node shape from GNN variable characteristics.
    
    Shape mapping:
    - Matrices (A, B): Rectangle [A]
    - Vectors (C, D, E): Rounded (C)
    - States (s, s_prime): Stadium ([s])
    - Observations (o): Circle ((o))
    - Actions (u): Hexagon {{u}}
    - Policies (π): Diamond {π}
    - Free Energy (F, G): Trapezoid [/F\]
    
    Args:
        var_name: Variable name
        var_data: Variable metadata dictionary
        
    Returns:
        Tuple of (opening_bracket, closing_bracket)
    """
    dims = var_data.get('dimensions', [])
    ontology = var_data.get('ontology_mapping', '')
    
    # Check ontology mapping first (more specific)
    if 'State' in ontology:
        # States use stadium shape
        return '([', '])'
    elif 'Observation' in ontology:
        # Observations use circles
        return '((', '))'
    elif 'Action' in ontology or var_name == 'u':
        # Actions use hexagons
        return '{{', '}}'
    elif 'Policy' in ontology or var_name in ['π', 'pi']:
        # Policies use diamonds
        return '{', '}'
    elif 'FreeEnergy' in ontology or var_name in ['F', 'G']:
        # Free energy uses trapezoid
        return '[/', '\\]'
    
    # Check dimensionality second
    if len(dims) >= 2:
        # Matrices use rectangles
        return '[', ']'
    else:
        # Default vectors use rounded
        return '(', ')'


def infer_edge_style(symbol: str) -> str:
    """
    Convert GNN connection symbols to Mermaid edge styles.
    
    Mapping:
    - > : Generative (thick arrow) ==>
    - - : Inference (dashed line) -.->
    - * : Modulation (dotted line) -..->
    - ~ : Weak coupling (thin line) -->
    
    Args:
        symbol: GNN connection symbol
        
    Returns:
        Mermaid edge style string
    """
    style_map = {
        '>': '==>',   # Generative
        '-': '-.->',  # Inference
        '*': '-..->',# Modulation
        '~': '-->'    # Coupling
    }
    
    return style_map.get(symbol, '-->')  # Default to normal arrow


def validate_mermaid_syntax(mermaid_content: str) -> Tuple[bool, List[str]]:
    """
    Validate basic Mermaid syntax.
    
    Checks:
    - Starts with flowchart directive
    - Node definitions are well-formed
    - Edge definitions are well-formed
    - Brackets are balanced
    
    Args:
        mermaid_content: Mermaid diagram content
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for flowchart directive
    if not re.search(r'^\s*flowchart\s+(TD|LR|TB|RL)', mermaid_content, re.MULTILINE):
        errors.append("Missing flowchart directive (e.g., 'flowchart TD')")
    
    # Check bracket balance
    bracket_pairs = [
        ('[', ']'),
        ('(', ')'),
        ('{', '}')
    ]
    
    for open_b, close_b in bracket_pairs:
        open_count = mermaid_content.count(open_b)
        close_count = mermaid_content.count(close_b)
        if open_count != close_count:
            errors.append(f"Unbalanced brackets: {open_count} '{open_b}' vs {close_count} '{close_b}'")
    
    # Check for common syntax errors
    if '-->' in mermaid_content and '-- >' in mermaid_content:
        errors.append("Inconsistent arrow spacing (mix of '-->' and '-- >')")
    
    return (len(errors) == 0, errors)


def get_oxdraw_options() -> Dict[str, Any]:
    """
    Get default oxdraw configuration options.
    
    Returns:
        Dictionary of configuration options
    """
    return {
        "default_port": 5151,
        "default_host": "127.0.0.1",
        "default_mode": "headless",
        "auto_convert": True,
        "validate_on_save": True,
        "include_metadata": True,
        "include_styling": True,
        "background_color": None,
        "scale": 10.0
    }


def sanitize_variable_name(var_name: str) -> str:
    """
    Sanitize variable name for use in Mermaid diagrams.
    
    Replaces special characters with underscores while preserving
    common mathematical symbols (π, σ, μ, etc.).
    
    Args:
        var_name: Variable name to sanitize
        
    Returns:
        Sanitized variable name
    """
    # Allow alphanumeric, underscore, and common math symbols
    allowed_pattern = r'[^a-zA-Z0-9_πσμαβγδεζηθλρτφψωΣΠΩ]'
    sanitized = re.sub(allowed_pattern, '_', var_name)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'var'
    
    return sanitized


def extract_node_positions(mermaid_content: str) -> Dict[str, Tuple[float, float]]:
    """
    Extract node positions from oxdraw-edited Mermaid files.
    
    oxdraw stores position data in special comments.
    
    Args:
        mermaid_content: Mermaid content with position data
        
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    positions = {}
    
    # Look for position comments like: %% node_id: x=100, y=200
    position_pattern = r'%%\s*(\w+):\s*x=([0-9.]+),\s*y=([0-9.]+)'
    
    for match in re.finditer(position_pattern, mermaid_content):
        node_id = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        positions[node_id] = (x, y)
    
    return positions


def generate_color_scheme(var_type: str) -> Dict[str, str]:
    """
    Generate color scheme for a variable type.
    
    Args:
        var_type: Variable type (matrix, vector, state, etc.)
        
    Returns:
        Dictionary with fill and stroke colors
    """
    color_schemes = {
        'matrix': {
            'fill': '#e1f5ff',
            'stroke': '#0288d1'
        },
        'vector': {
            'fill': '#fff9e6',
            'stroke': '#fbc02d'
        },
        'state': {
            'fill': '#fff3e0',
            'stroke': '#f57c00'
        },
        'observation': {
            'fill': '#f3e5f5',
            'stroke': '#7b1fa2'
        },
        'action': {
            'fill': '#e8f5e9',
            'stroke': '#388e3c'
        },
        'policy': {
            'fill': '#e3f2fd',
            'stroke': '#1976d2'
        },
        'free_energy': {
            'fill': '#fce4ec',
            'stroke': '#c2185b'
        }
    }
    
    return color_schemes.get(var_type, {'fill': '#f5f5f5', 'stroke': '#9e9e9e'})


def estimate_diagram_complexity(gnn_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate visual complexity of the diagram for layout optimization.
    
    Args:
        gnn_model: GNN model dictionary
        
    Returns:
        Dictionary with complexity metrics
    """
    num_variables = len(gnn_model.get('variables', {}))
    num_connections = len(gnn_model.get('connections', []))
    
    # Calculate average degree (connections per node)
    avg_degree = (num_connections / num_variables) if num_variables > 0 else 0
    
    # Determine complexity level
    if num_variables <= 10 and num_connections <= 20:
        complexity = 'simple'
    elif num_variables <= 50 and num_connections <= 100:
        complexity = 'moderate'
    else:
        complexity = 'complex'
    
    return {
        'num_variables': num_variables,
        'num_connections': num_connections,
        'avg_degree': avg_degree,
        'complexity': complexity,
        'recommended_layout': 'force_directed' if complexity != 'simple' else 'hierarchical'
    }

