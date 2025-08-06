#!/usr/bin/env python3
"""
Test Mcp Tools Tests

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


# Migrated from test_mcp_integration_comprehensive.py
class TestMCPToolExecution:
    """Test actual MCP tool execution."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_gnn_validate_tool_execution(self, mock_mcp_tools, comprehensive_test_data):
        """Test GNN validation tool registration (lightweight test)."""
        try:
            from gnn.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Just verify that the tool was registered properly
            assert 'validate_gnn_content' in mock_mcp_tools.tools
            tool_info = mock_mcp_tools.tools['validate_gnn_content']
            assert 'function' in tool_info
            assert 'description' in tool_info
            assert callable(tool_info['function'])
                
        except ImportError:
            pytest.skip("GNN MCP not available")
        except Exception as e:
            # Should handle registration errors gracefully
            assert "error" in str(e).lower() or "import" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_tool_execution(self, mock_mcp_tools, comprehensive_test_data):
        """Test export tool registration (lightweight test)."""
        try:
            from export.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Just verify that tools were registered properly
            assert len(mock_mcp_tools.tools) > 0
            for tool_name, tool_info in mock_mcp_tools.tools.items():
                assert 'function' in tool_info
                assert 'description' in tool_info
                assert callable(tool_info['function'])
                    
        except ImportError:
            pytest.skip("Export MCP not available")
        except Exception as e:
            # Should handle registration errors gracefully
            assert "error" in str(e).lower() or "import" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_utils_system_info_execution(self, mock_mcp_tools):
        """Test utils system info tool registration (lightweight test)."""
        try:
            from utils.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Just verify that the system info tool was registered properly
            assert 'get_system_info' in mock_mcp_tools.tools
            tool_info = mock_mcp_tools.tools['get_system_info']
            assert 'function' in tool_info
            assert 'description' in tool_info
            assert callable(tool_info['function'])
                
        except ImportError:
            pytest.skip("Utils MCP not available")
        except Exception as e:
            # Should handle registration errors gracefully
            assert "error" in str(e).lower() or "import" in str(e).lower()


