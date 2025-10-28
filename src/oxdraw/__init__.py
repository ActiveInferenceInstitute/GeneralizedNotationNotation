"""
oxdraw Integration Module for GNN Pipeline

Provides visual diagram-as-code interface for Active Inference model construction
through bidirectional GNN ↔ Mermaid ↔ oxdraw synchronization.

Public API:
- process_oxdraw: Main processing function for pipeline integration
- gnn_to_mermaid: Convert GNN models to Mermaid format
- mermaid_to_gnn: Parse Mermaid diagrams back to GNN
- convert_gnn_file_to_mermaid: File-based GNN → Mermaid conversion
- convert_mermaid_file_to_gnn: File-based Mermaid → GNN conversion
- check_oxdraw_installed: Verify oxdraw CLI availability
- launch_oxdraw_editor: Launch interactive oxdraw editor
"""

from pathlib import Path
from typing import Dict, Any, Optional

# Import core processing
from .processor import (
    process_oxdraw,
    check_oxdraw_installed,
    launch_oxdraw_editor,
    get_module_info
)

# Import converters
from .mermaid_converter import (
    gnn_to_mermaid,
    convert_gnn_file_to_mermaid,
    generate_mermaid_metadata
)

# Import parser
from .mermaid_parser import (
    mermaid_to_gnn,
    convert_mermaid_file_to_gnn,
    extract_gnn_metadata
)

# Import utilities
from .utils import (
    infer_node_shape,
    infer_edge_style,
    validate_mermaid_syntax,
    get_oxdraw_options
)

__version__ = "1.0.0"

FEATURES = {
    "gnn_to_mermaid": True,
    "mermaid_to_gnn": True,
    "interactive_editor": True,
    "headless_conversion": True,
    "ontology_integration": True,
    "validation": True,
    "mcp_integration": True
}

# Feature availability checks
def check_features() -> Dict[str, bool]:
    """Check availability of optional features."""
    features = FEATURES.copy()
    
    # Check oxdraw CLI availability
    features["oxdraw_cli"] = check_oxdraw_installed()
    
    return features

__all__ = [
    # Core processing
    'process_oxdraw',
    'check_oxdraw_installed',
    'launch_oxdraw_editor',
    'get_module_info',
    
    # Converters
    'gnn_to_mermaid',
    'convert_gnn_file_to_mermaid',
    'generate_mermaid_metadata',
    
    # Parser
    'mermaid_to_gnn',
    'convert_mermaid_file_to_gnn',
    'extract_gnn_metadata',
    
    # Utilities
    'infer_node_shape',
    'infer_edge_style',
    'validate_mermaid_syntax',
    'get_oxdraw_options',
    
    # Module info
    '__version__',
    'FEATURES',
    'check_features'
]

