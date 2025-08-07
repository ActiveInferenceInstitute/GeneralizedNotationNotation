"""
Export module for GNN Processing Pipeline.

This module provides multi-format export capabilities for GNN files.
"""

from .processor import (
    generate_exports,
    export_single_gnn_file,
    parse_gnn_content,
    export_model,
    _gnn_model_to_dict,
    export_gnn_model
)

from .formatters import (
    export_to_json,
    export_to_xml,
    export_to_graphml,
    export_to_gexf,
    export_to_pickle,
    export_to_json_gnn,
    export_to_xml_gnn,
    export_to_python_pickle,
    export_to_plaintext_summary,
    export_to_plaintext_dsl
)

from .utils import (
    get_module_info,
    get_supported_formats
)

__version__ = "1.0.0"
FEATURES = {"json_export": True, "xml_export": True, "graphml_export": True, "gexf_export": True, "pickle_export": True}
HAS_NETWORKX = True

__all__ = [
    'generate_exports',
    'export_single_gnn_file',
    'parse_gnn_content',
    'export_model',
    'export_gnn_model',
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
    '__version__',
    'FEATURES',
    'HAS_NETWORKX'
]
