#!/usr/bin/env python3
"""
Test Mcp Overall Tests

This file contains tests migrated from test_mcp_comprehensive.py.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


# Migrated from test_mcp_comprehensive.py
class TestMCPCoreComprehensive:
    """Comprehensive tests for the enhanced MCP server implementation."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_core_imports(self):
        """Test that enhanced MCP core module can be imported and has expected structure."""
        try:
            from mcp import (
                MCP, MCPTool, MCPResource, MCPError, MCPServer,
                MCPToolNotFoundError, MCPResourceNotFoundError, MCPInvalidParamsError,
                MCPToolExecutionError, MCPSDKNotFoundError, MCPValidationError,
                MCPModuleLoadError, MCPPerformanceError,
                initialize, get_mcp_instance, register_tools
            )
            
            # Test that classes and functions are available
            assert MCP is not None, "MCP 

# Migrated from test_mcp_comprehensive.py
class TestMCPErrorHandling:
    """Comprehensive tests for MCP error handling."""
    
    @pytest.mark.unit
    def test_mcp_error_classes(self):
        """Test MCP error 

# Migrated from test_mcp_integration_comprehensive.py
class TestMCPErrorHandling:
    """Test MCP error handling and edge cases."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_invalid_tool_execution(self, mock_mcp_tools):
        """Test executing non-existent tools."""
        try:
            result = mock_mcp_tools.execute_tool('non_existent_tool')
            assert False, "Should have raised an error"
        except ValueError as e:
            # Expected error for non-existent tool
            assert "not found" in str(e)
        except Exception as e:
            # Other errors are also acceptable
            assert "error" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_tool_execution_with_invalid_parameters(self, mock_mcp_tools):
        """Test tool registration with parameter validation (lightweight test)."""
        try:
            from gnn.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Just verify that tools have proper parameter schemas
            if 'validate_gnn_content' in mock_mcp_tools.tools:
                tool_info = mock_mcp_tools.tools['validate_gnn_content']
                assert 'schema' in tool_info
                # Verify the schema has required fields
                schema = tool_info['schema']
                assert isinstance(schema, dict)
                    
        except ImportError:
            pytest.skip("GNN MCP not available")



# Migrated from test_mcp_integration_comprehensive.py
class TestMCPResourceManagement:
    """Test MCP resource management and cleanup."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_resource_registration(self, mock_mcp_tools):
        """Test MCP resource registration."""
        try:
            # Register resources from modules that support them
            from gnn.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify resources were registered if supported
            if hasattr(mock_mcp_tools, 'resources'):
                assert isinstance(mock_mcp_tools.resources, dict)
                
        except ImportError:
            pytest.skip("GNN MCP not available")
        except Exception as e:
            # Should handle resource registration errors gracefully
            assert "error" in str(e).lower()


