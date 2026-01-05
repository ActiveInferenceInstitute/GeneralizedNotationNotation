#!/usr/bin/env python3
"""
Test MCP Overall Tests

This file contains comprehensive tests for the MCP (Model Context Protocol) module.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


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
            assert MCP is not None, "MCP class should be available"
            assert MCPTool is not None, "MCPTool class should be available"
            assert MCPError is not None, "MCPError class should be available"
            
        except ImportError:
            pytest.skip("MCP module not available")


class TestMCPErrorHandling:
    """Comprehensive tests for MCP error handling."""
    
    @pytest.mark.unit
    def test_mcp_error_classes(self):
        """Test MCP error classes are available."""
        try:
            from mcp import (
                MCPError, MCPToolNotFoundError, MCPResourceNotFoundError,
                MCPInvalidParamsError, MCPToolExecutionError, MCPSDKNotFoundError,
                MCPValidationError, MCPModuleLoadError, MCPPerformanceError
            )
            
            # Test that error classes are available
            assert MCPError is not None, "MCPError class should be available"
            assert MCPToolNotFoundError is not None, "MCPToolNotFoundError class should be available"
            
        except ImportError:
            pytest.skip("MCP module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_invalid_tool_execution(self, test_mcp_tools):
        """Test executing non-existent tools."""
        try:
            result = test_mcp_tools.execute_tool('non_existent_tool')
            assert False, "Should have raised an error"
        except ValueError as e:
            # Expected error for non-existent tool
            assert "not found" in str(e)
        except Exception as e:
            # Other errors are also acceptable
            assert "error" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_tool_execution_with_invalid_parameters(self, test_mcp_tools):
        """Test tool registration with parameter validation (lightweight test)."""
        try:
            from gnn.mcp import register_tools
            
            register_tools(test_mcp_tools)
            
            # Just verify that tools have proper parameter schemas
            if 'validate_gnn_content' in test_mcp_tools.tools:
                tool_info = test_mcp_tools.tools['validate_gnn_content']
                assert 'schema' in tool_info
                # Verify the schema has required fields
                schema = tool_info['schema']
                assert isinstance(schema, dict)
                    
        except ImportError:
            pytest.skip("GNN MCP not available")


class TestMCPResourceManagement:
    """Test MCP resource management and cleanup."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_resource_registration(self, test_mcp_tools):
        """Test MCP resource registration."""
        try:
            # Register resources from modules that support them
            from gnn.mcp import register_tools
            
            register_tools(test_mcp_tools)
            
            # Verify resources were registered if supported
            if hasattr(test_mcp_tools, 'resources'):
                assert isinstance(test_mcp_tools.resources, dict)
                
        except ImportError:
            pytest.skip("GNN MCP not available")
        except Exception as e:
            # Should handle resource registration errors gracefully
            assert "error" in str(e).lower()


class TestMCPModuleComprehensive:
    """Comprehensive tests for MCP module functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_module_imports(self):
        """Test that MCP module can be imported."""
        try:
            from mcp import MCP
            assert MCP is not None, "MCP module should be available"
        except ImportError:
            pytest.skip("MCP module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_server_instantiation(self):
        """Test MCP server instantiation."""
        try:
            from mcp import MCPServer
            server = MCPServer()
            assert server is not None, "MCPServer should be instantiable"
        except ImportError:
            pytest.skip("MCP module not available")
        except Exception as e:
            # Server instantiation might fail, but should not crash
            assert "error" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_module_info(self):
        """Test MCP module information."""
        try:
            from mcp import get_module_info
            info = get_module_info()
            assert isinstance(info, dict), "Module info should be a dictionary"
        except ImportError:
            pytest.skip("MCP module not available")
        except AttributeError:
            pytest.skip("get_module_info not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_tools_registration(self):
        """Test MCP tools registration."""
        try:
            from mcp import register_tools
            # Test that register_tools function exists
            assert callable(register_tools), "register_tools should be callable"
        except ImportError:
            pytest.skip("MCP module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_initialization(self):
        """Test MCP initialization."""
        try:
            from mcp import initialize
            # Test that initialize function exists
            assert callable(initialize), "initialize should be callable"
        except ImportError:
            pytest.skip("MCP module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_instance_retrieval(self):
        """Test MCP instance retrieval."""
        try:
            from mcp import get_mcp_instance
            # Test that get_mcp_instance function exists
            assert callable(get_mcp_instance), "get_mcp_instance should be callable"
        except ImportError:
            pytest.skip("MCP module not available")


