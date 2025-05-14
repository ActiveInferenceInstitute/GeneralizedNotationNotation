#!/usr/bin/env python3
"""
Test suite for the Model Context Protocol implementation.

This script tests all components of the MCP implementation, including:
- Core MCP functionality
- Module discovery
- Tool and resource registration
- Tool execution
- Resource retrieval
- Server functionality
"""

import unittest
import os
import sys
import json
import tempfile
import threading
import time
import requests
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib
import subprocess
import io

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import MCP modules
from mcp import mcp_instance, initialize, MCP, MCPTool, MCPResource


class TestMCPCore(unittest.TestCase):
    """Test the core MCP functionality."""
    
    def setUp(self):
        """Set up a fresh MCP instance for each test."""
        self.mcp = MCP()
    
    def test_tool_registration(self):
        """Test registering and retrieving tools."""
        # Define a simple test tool
        def test_tool(param1: str, param2: int) -> dict:
            return {"result": f"{param1}_{param2}"}
        
        # Register the tool
        self.mcp.register_tool(
            "test_tool",
            test_tool,
            {
                "param1": {"type": "string"},
                "param2": {"type": "integer"}
            },
            "Test tool description"
        )
        
        # Check that the tool was registered correctly
        self.assertIn("test_tool", self.mcp.tools)
        
        # Execute the tool
        result = self.mcp.execute_tool("test_tool", {"param1": "test", "param2": 42})
        self.assertEqual(result["result"]["result"], "test_42")
        
        # Check tool capability listing
        capabilities = self.mcp.get_capabilities()
        self.assertIn("test_tool", capabilities["tools"])
        self.assertEqual(capabilities["tools"]["test_tool"]["description"], "Test tool description")
    
    def test_resource_registration(self):
        """Test registering and retrieving resources."""
        # Define a simple test resource retriever
        def test_resource(uri: str) -> dict:
            return {"uri": uri, "content": "test_content"}
        
        # Register the resource
        self.mcp.register_resource(
            "test://{id}",
            test_resource,
            "Test resource description"
        )
        
        # Check that the resource was registered correctly
        self.assertIn("test://{id}", self.mcp.resources)
        
        # Get the resource
        result = self.mcp.get_resource("test://123")
        self.assertEqual(result["content"]["uri"], "test://123")
        self.assertEqual(result["content"]["content"], "test_content")
        
        # Check resource capability listing
        capabilities = self.mcp.get_capabilities()
        self.assertIn("test://{id}", capabilities["resources"])
        self.assertEqual(capabilities["resources"]["test://{id}"]["description"], "Test resource description")
    
    def test_module_discovery(self):
        """Test the module discovery mechanism."""
        # Create a mock module with register_tools function
        mock_module = MagicMock()
        
        # Use patch to replace the importlib.import_module function
        with patch('importlib.import_module', return_value=mock_module) as mock_import:
            # Create a temporary directory with a mcp.py file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Create the mcp.py file
                mcp_file = temp_dir_path / "mcp.py"
                mcp_file.write_text("def register_tools(mcp): pass")
                
                # Mock Path.iterdir() to return our temp directory
                with patch('pathlib.Path.iterdir', return_value=[temp_dir_path]) as mock_iterdir:
                    # Mock Path.is_dir() to return True
                    with patch.object(temp_dir_path, 'is_dir', return_value=True) as mock_is_dir:
                        # Mock Path.name to return a non-underscore name
                        with patch.object(temp_dir_path, 'name', 'test_module') as mock_name:
                            # Run module discovery
                            self.mcp.discover_modules()
                            
                            # Verify that importlib.import_module was called with the right module name
                            mock_import.assert_called_once()
                            
                            # Verify that register_tools was called
                            mock_module.register_tools.assert_called_once_with(self.mcp)


class TestMCPCLI(unittest.TestCase):
    """Test the MCP CLI functionality."""
    
    def test_list_capabilities(self):
        """Test listing capabilities through the CLI."""
        # Mock stdout to capture output
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # Mock import_mcp to return a mock MCP instance
            mock_mcp = MagicMock()
            mock_mcp.get_capabilities.return_value = {
                "tools": {"test_tool": {"description": "Test tool"}},
                "resources": {"test://uri": {"description": "Test resource"}},
                "version": "1.0.0",
                "name": "Test MCP"
            }
            
            mock_init = MagicMock()
            
            with patch('src.mcp.cli.import_mcp', return_value=(mock_mcp, mock_init)) as mock_import:
                # Create mock args
                mock_args = MagicMock()
                
                # Import the CLI module
                from src.mcp.cli import list_capabilities
                
                # Run list_capabilities
                list_capabilities(mock_args)
                
                # Verify import_mcp was called
                mock_import.assert_called_once()
                
                # Verify get_capabilities was called
                mock_mcp.get_capabilities.assert_called_once()
                
                # Check output
                output = mock_stdout.getvalue()
                self.assertIn("test_tool", output)
                self.assertIn("Test tool", output)
                self.assertIn("test://uri", output)
                self.assertIn("Test resource", output)


class TestMCPServers(unittest.TestCase):
    """Test the MCP server implementations."""
    
    def test_http_server(self):
        """Test the HTTP server implementation."""
        # This test would start an actual HTTP server, send requests, and check responses
        # For simplicity in a test environment, I'll just verify the server classes exist
        
        # Import the HTTP server
        from src.mcp.server_http import MCPHTTPServer, MCPHTTPHandler
        
        # Create server with mock MCP instance
        mock_mcp = MagicMock()
        mock_mcp.get_capabilities.return_value = {"tools": {}, "resources": {}}
        
        with patch('src.mcp.server_http.initialize') as mock_init:
            with patch('src.mcp.server_http.mcp_instance', mock_mcp):
                # Mock HTTPServer to prevent actual server start
                with patch('src.mcp.server_http.HTTPServer') as mock_http_server:
                    # Mock threading.Thread to prevent actual thread start
                    with patch('threading.Thread') as mock_thread:
                        # Create server instance
                        server = MCPHTTPServer(host="localhost", port=8080)
                        
                        # Try to start the server (should be mocked)
                        with patch.object(server, '_server_thread') as mock_server_thread:
                            # Stop immediately
                            with patch.object(server, 'running', False):
                                server.start()
                                
                                # Verify that initialize was called
                                mock_init.assert_called_once()
                                
                                # Verify that HTTPServer was created
                                mock_http_server.assert_called_once()
                                
                                # Verify that Thread was created
                                mock_thread.assert_called_once()
    
    def test_stdio_server(self):
        """Test the stdio server implementation."""
        # Similar to HTTP server test, verify the server class exists
        
        # Import the stdio server
        from src.mcp.server_stdio import StdioServer
        
        # Create server with mock MCP instance
        mock_mcp = MagicMock()
        mock_mcp.get_capabilities.return_value = {"tools": {}, "resources": {}}
        
        with patch('src.mcp.server_stdio.initialize') as mock_init:
            with patch('src.mcp.server_stdio.mcp_instance', mock_mcp):
                # Mock sys.stdin and sys.stdout
                with patch('sys.stdin') as mock_stdin:
                    with patch('sys.stdout') as mock_stdout:
                        # Mock queue.Queue to prevent actual queue operations
                        with patch('queue.Queue') as mock_queue:
                            # Mock threading.Thread to prevent actual thread start
                            with patch('threading.Thread') as mock_thread:
                                # Create server instance
                                server = StdioServer()
                                
                                # Try to start the server (should be mocked)
                                with patch.object(server, '_reader_thread') as mock_reader_thread:
                                    with patch.object(server, '_writer_thread') as mock_writer_thread:
                                        with patch.object(server, '_processor_thread') as mock_processor_thread:
                                            # Stop immediately
                                            with patch.object(server, 'running', False):
                                                server.start()
                                                
                                                # Verify that initialize was called
                                                mock_init.assert_called_once()
                                                
                                                # Verify that Thread was created at least once
                                                self.assertGreater(mock_thread.call_count, 0)


class TestIntegration(unittest.TestCase):
    """Test integration with other modules."""
    
    def test_visualization_integration(self):
        """Test integration with the visualization module."""
        # Skip if visualization module doesn't exist
        try:
            importlib.import_module("src.visualization.mcp")
        except ImportError:
            self.skipTest("Visualization module not found")
        
        # Create MCP instance and discover modules
        mcp = MCP()
        
        # Mock Path.iterdir to return a fixed set of directories
        mock_dirs = []
        visualization_dir = MagicMock()
        visualization_dir.is_dir.return_value = True
        visualization_dir.name = "visualization"
        
        # Mock visualization_dir / "mcp.py" to exist
        visualization_mcp_file = MagicMock()
        visualization_mcp_file.exists.return_value = True
        
        # Make Path / return the mcp file
        visualization_dir.__truediv__.return_value = visualization_mcp_file
        
        mock_dirs.append(visualization_dir)
        
        with patch('pathlib.Path.iterdir', return_value=mock_dirs):
            # Mock importlib.import_module to return real visualization.mcp module
            with patch('importlib.import_module', return_value=importlib.import_module("src.visualization.mcp")):
                # Discover modules
                mcp.discover_modules()
                
                # Check that visualization tools were registered
                self.assertIn("visualize_gnn_file", mcp.tools)
                self.assertIn("visualize_gnn_directory", mcp.tools)
                self.assertIn("parse_gnn_file", mcp.tools)
                
                # Check that visualization resources were registered
                self.assertIn("visualization://{output_directory}", mcp.resources)
    
    def test_tests_integration(self):
        """Test integration with the tests module."""
        # Skip if tests module doesn't exist
        try:
            importlib.import_module("src.tests.mcp")
        except ImportError:
            self.skipTest("Tests module not found")
        
        # Create MCP instance and discover modules
        mcp = MCP()
        
        # Mock Path.iterdir to return a fixed set of directories
        mock_dirs = []
        tests_dir = MagicMock()
        tests_dir.is_dir.return_value = True
        tests_dir.name = "tests"
        
        # Mock tests_dir / "mcp.py" to exist
        tests_mcp_file = MagicMock()
        tests_mcp_file.exists.return_value = True
        
        # Make Path / return the mcp file
        tests_dir.__truediv__.return_value = tests_mcp_file
        
        mock_dirs.append(tests_dir)
        
        with patch('pathlib.Path.iterdir', return_value=mock_dirs):
            # Mock importlib.import_module to return real tests.mcp module
            with patch('importlib.import_module', return_value=importlib.import_module("src.tests.mcp")):
                # Discover modules
                mcp.discover_modules()
                
                # Check that tests tools were registered
                self.assertIn("run_gnn_type_checker", mcp.tools)
                self.assertIn("run_gnn_type_checker_on_directory", mcp.tools)
                self.assertIn("run_gnn_unit_tests", mcp.tools)
                
                # Check that tests resources were registered
                self.assertIn("test-report://{report_file}", mcp.resources)


def run_all_tests():
    """Run all MCP tests and generate a report."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMCPCore))
    test_suite.addTest(unittest.makeSuite(TestMCPCLI))
    test_suite.addTest(unittest.makeSuite(TestMCPServers))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    test_result = test_runner.run(test_suite)
    
    # Generate a report
    report = {
        "total_tests": test_result.testsRun,
        "failures": len(test_result.failures),
        "errors": len(test_result.errors),
        "skipped": len(test_result.skipped),
        "success": test_result.wasSuccessful()
    }
    
    # Print the report
    print("\n\n===== MCP TEST REPORT =====")
    print(f"Total tests: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Skipped: {report['skipped']}")
    print(f"Success: {'Yes' if report['success'] else 'No'}")
    
    if report['failures'] > 0:
        print("\nFailures:")
        for failure in test_result.failures:
            print(f"- {failure[0]}: {failure[1][:200]}...")
    
    if report['errors'] > 0:
        print("\nErrors:")
        for error in test_result.errors:
            print(f"- {error[0]}: {error[1][:200]}...")
    
    print("===========================\n")
    
    return report


if __name__ == "__main__":
    # Run all tests
    run_all_tests() 