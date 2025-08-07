#!/usr/bin/env python3
"""
Export utils module for GNN Processing Pipeline.

This module provides export utility functions.
"""

from typing import Dict, Any, List

def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the export module and its capabilities."""
    info = {
        'version': "1.0.0",
        'description': "Multi-format export capabilities for GNN Processing Pipeline",
        'features': {
            'json_export': True,
            'xml_export': True,
            'graphml_export': True,
            'gexf_export': True,
            'pickle_export': True,
            'plaintext_export': True,
            'dsl_export': True
        },
        'export_capabilities': [],
        'supported_formats': [],
        'export_methods': []
    }
    
    # Export capabilities
    info['export_capabilities'].extend([
        'JSON export',
        'XML export',
        'GraphML export',
        'GEXF export',
        'Pickle export',
        'Plaintext summary',
        'DSL export'
    ])
    
    # Export methods
    info['export_methods'].extend([
        'Single file export',
        'Batch export',
        'Format-specific export',
        'Model data export'
    ])
    
    # Supported formats
    info['supported_formats'].extend(['json', 'xml', 'graphml', 'gexf', 'pkl', 'txt', 'dsl'])
    
    return info

def get_supported_formats() -> Dict[str, List[str]]:
    """Get information about supported export formats."""
    return {
        'data_formats': ['json', 'xml', 'pickle'],
        'graph_formats': ['graphml', 'gexf'],
        'text_formats': ['txt', 'dsl'],
        'all_formats': ['json', 'xml', 'graphml', 'gexf', 'pkl', 'txt', 'dsl']
    }
