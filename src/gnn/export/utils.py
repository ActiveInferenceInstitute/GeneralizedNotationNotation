#!/usr/bin/env python3
"""
Export utils module for GNN Processing Pipeline.

This module provides export utility functions.
"""

from typing import Any, Dict, List

from .registry import get_export_registry, get_format_categories


def get_module_info() -> Dict[str, Any]:
    """Get comprehensive information about the export module and its capabilities."""
    formats = list(get_export_registry().keys())
    categories = get_format_categories()
    return {
        "version": "1.0.0",
        "description": "Multi-format export capabilities for GNN Processing Pipeline",
        "features": {
            "json_export": True,
            "xml_export": True,
            "graphml_export": True,
            "gexf_export": True,
            "pickle_export": True,
            "plaintext_export": True,
            "dsl_export": True,
        },
        "export_capabilities": [
            "JSON export",
            "XML export",
            "GraphML export",
            "GEXF export",
            "Pickle export",
            "Plaintext summary",
            "DSL export",
        ],
        "supported_formats": formats,
        "export_methods": [
            "Single file export",
            "Batch export",
            "Format-specific export",
            "Model data export",
        ],
        "available_formats": formats,
        "graph_formats": categories["graph"],
        "text_formats": categories["text"],
        "data_formats": categories["data"],
    }


def get_supported_formats() -> Dict[str, List[str]]:
    """Get information about supported export formats, grouped by category.

    Keys: ``data_formats``, ``graph_formats``, ``text_formats``,
    ``all_formats``. Derived from :mod:`export.registry` so it cannot drift
    from the dispatch tables.
    """
    categories = get_format_categories()
    return {
        "data_formats": categories["data"],
        "graph_formats": categories["graph"],
        "text_formats": categories["text"],
        "all_formats": list(get_export_registry().keys()),
    }
