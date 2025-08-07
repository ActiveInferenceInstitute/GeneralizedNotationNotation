#!/usr/bin/env python3
"""
Ontology utils module for GNN Processing Pipeline.

This module provides ontology utility functions.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

def get_module_info() -> Dict[str, Any]:
    """
    Get module information for ontology processing.
    
    Returns:
        Dictionary with module information
    """
    return {
        "module_name": "ontology",
        "description": "Ontology processing for GNN files",
        "version": "1.0.0",
        "author": "GNN Team",
        "capabilities": [
            "ontology_parsing",
            "ontology_validation", 
            "ontology_mapping",
            "ontology_reporting"
        ],
        "supported_formats": ["markdown", "json"],
        "dependencies": ["pathlib", "typing", "logging", "json"]
    }

def get_ontology_processing_options() -> Dict[str, Any]:
    """
    Get ontology processing options and configuration.
    
    Returns:
        Dictionary with processing options
    """
    return {
        "strict_validation": True,
        "include_metadata": True,
        "generate_reports": True,
        "output_formats": ["json", "markdown"],
        "validation_rules": [
            "required_terms",
            "ontology_compliance",
            "semantic_consistency"
        ],
        "processing_modes": [
            "basic",
            "comprehensive", 
            "validation_only"
        ]
    }

def get_mcp_interface() -> Dict[str, Any]:
    """
    Get MCP interface configuration for ontology module.
    
    Returns:
        Dictionary with MCP interface configuration
    """
    return {
        "tools": [
            {
                "name": "process_ontology",
                "description": "Process ontology for GNN files",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target_dir": {"type": "string"},
                        "output_dir": {"type": "string"},
                        "verbose": {"type": "boolean"}
                    }
                }
            },
            {
                "name": "parse_gnn_ontology_section",
                "description": "Parse GNN ontology section from content",
                "inputSchema": {
                    "type": "object", 
                    "properties": {
                        "content": {"type": "string"}
                    }
                }
            },
            {
                "name": "process_gnn_ontology",
                "description": "Process ontology for a single GNN file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "gnn_file": {"type": "string"}
                    }
                }
            },
            {
                "name": "validate_annotations",
                "description": "Validate annotations against ontology terms",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "annotations": {"type": "array", "items": {"type": "string"}},
                        "ontology_terms": {"type": "object"}
                    }
                }
            }
        ],
        "resources": [
            {
                "name": "ontology_terms.json",
                "description": "Defined ontology terms",
                "type": "file"
            }
        ]
    }
