"""
Type checker module for GNN Processing Pipeline.

This module provides comprehensive GNN syntax validation, type checking,
resource estimation, and MCP integration for the Generalized Notation Notation
processing pipeline.

Key Features:
- GNN file validation and syntax checking
- Variable type analysis and dimension validation
- Connection pattern analysis and complexity estimation
- Computational resource estimation
- Performance analysis and optimization suggestions
- MCP (Model Context Protocol) integration
- Comprehensive error handling and logging

Architecture:
- processor.py: Core type checking functionality
- analysis_utils.py: Analysis utilities and complexity estimation
- mcp.py: MCP tool registration and execution
"""

from .processor import GNNTypeChecker, estimate_file_resources, process_type_checking_standardized
from .analysis_utils import (
    analyze_variable_types,
    analyze_connections,
    estimate_computational_complexity
)
from .mcp import (
    register_mcp_tools,
    execute_mcp_tool,
    get_mcp_tool_schema,
    list_available_tools,
    validate_tool_arguments
)

__version__ = "1.0.0"
__author__ = "GNN Processing Pipeline Team"
FEATURES = {
    "gnn_syntax_validation": True,
    "variable_type_analysis": True,
    "connection_analysis": True,
    "resource_estimation": True,
    "complexity_estimation": True,
    "performance_analysis": True,
    "mcp_integration": True,
    "error_handling": True
}

__all__ = [
    # Core functionality
    'GNNTypeChecker',
    'estimate_file_resources',
    'process_type_checking_standardized',

    # Analysis utilities
    'analyze_variable_types',
    'analyze_connections',
    'estimate_computational_complexity',

    # MCP integration
    'register_mcp_tools',
    'execute_mcp_tool',
    'get_mcp_tool_schema',
    'list_available_tools',
    'validate_tool_arguments',

    # Metadata
    '__version__',
    'FEATURES',
    '__author__'
] 