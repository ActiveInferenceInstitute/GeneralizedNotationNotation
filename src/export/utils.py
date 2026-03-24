#!/usr/bin/env python3
"""
Export utils module for GNN Processing Pipeline.

This module provides export utility functions.
"""

from typing import Any, Dict, List


def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the export module and its capabilities."""
    return {
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
        'export_capabilities': [
            'JSON export',
            'XML export',
            'GraphML export',
            'GEXF export',
            'Pickle export',
            'Plaintext summary',
            'DSL export',
        ],
        'supported_formats': ['json', 'xml', 'graphml', 'gexf', 'pickle', 'txt', 'dsl'],
        'export_methods': [
            'Single file export',
            'Batch export',
            'Format-specific export',
            'Model data export',
        ],
        'available_formats': ['json', 'xml', 'graphml', 'gexf', 'pickle', 'txt', 'dsl'],
        'graph_formats': ['graphml', 'gexf'],
        'text_formats': ['txt', 'dsl'],
        'data_formats': ['json', 'xml', 'pickle'],
    }

def get_supported_formats() -> Dict[str, List[str]]:
    """Get information about supported export formats."""
    return {
        'data_formats': ['json', 'xml', 'pickle'],
        'graph_formats': ['graphml', 'gexf'],
        'text_formats': ['txt', 'dsl'],
        'all_formats': ['json', 'xml', 'graphml', 'gexf', 'pickle', 'txt', 'dsl']
    }
