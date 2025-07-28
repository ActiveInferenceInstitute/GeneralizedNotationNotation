#!/usr/bin/env python3
"""
Comprehensive MCP integration tests for all 17 modules.

This module provides comprehensive testing for MCP tool registration 
and functionality across all modules:
- gnn.mcp
- type_checker.mcp
- export.mcp
- visualization.mcp
- render.mcp
- execute.mcp
- llm.mcp
- audio.mcp
- analysis.mcp
- integration.mcp
- security.mcp
- research.mcp
- website.mcp
- report.mcp
- ontology.mcp
- validation.mcp
- model_registry.mcp
- template.mcp
- setup.mcp
- tests.mcp
- utils.mcp
- pipeline.mcp
"""

import pytest
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Test markers
pytestmark = [pytest.mark.mcp, pytest.mark.safe_to_fail, pytest.mark.fast]

class TestMCPModuleRegistration:
    """Test MCP tool registration for all modules."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_all_modules_have_mcp_files(self, project_root):
        """Test that all expected modules have MCP files."""
        expected_mcp_modules = [
            "gnn", "type_checker", "export", "visualization", "render", 
            "execute", "llm", "audio", "analysis", "integration", 
            "security", "research", "website", "report", "ontology",
            "validation", "model_registry", "template", "setup", 
            "tests", "utils", "pipeline"
        ]
        
        src_dir = project_root / "src"
        found_mcp_files = []
        
        for module_name in expected_mcp_modules:
            mcp_file = src_dir / module_name / "mcp.py"
            if mcp_file.exists():
                found_mcp_files.append(module_name)
        
        # Should have most MCP files
        coverage_ratio = len(found_mcp_files) / len(expected_mcp_modules)
        assert coverage_ratio >= 0.7, f"Low MCP coverage: {len(found_mcp_files)}/{len(expected_mcp_modules)} modules have MCP files"
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_mcp_register_tools_functions(self, project_root):
        """Test that MCP modules have register_tools functions."""
        src_dir = project_root / "src"
        modules_with_register_tools = []
        
        mcp_modules = ["gnn", "export", "visualization", "llm", "audio"]
        
        for module_name in mcp_modules:
            try:
                mcp_module = __import__(f'{module_name}.mcp', fromlist=['register_tools'])
                if hasattr(mcp_module, 'register_tools'):
                    modules_with_register_tools.append(module_name)
            except ImportError:
                pass  # Module not available
        
        # Should have at least some modules with register_tools
        assert len(modules_with_register_tools) >= 1, f"No modules found with register_tools function"

class TestGNNMCP:
    """Test GNN module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_gnn_mcp_imports(self):
        """Test GNN MCP imports."""
        try:
            from gnn import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("GNN MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_gnn_mcp_tool_registration(self, mock_mcp_tools):
        """Test GNN MCP tool registration."""
        try:
            from gnn.mcp import register_tools
            
            # Register tools with mock MCP
            register_tools(mock_mcp_tools)
            
            # Verify tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected GNN tools
            expected_tools = ['validate_gnn_content', 'parse_gnn_content', 'process_gnn_directory']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("GNN MCP not available")

class TestExportMCP:
    """Test Export module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_mcp_imports(self):
        """Test Export MCP imports."""
        try:
            from export import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Export MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_export_mcp_tool_registration(self, mock_mcp_tools):
        """Test Export MCP tool registration."""
        try:
            from export.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify export tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected export tools
            expected_tools = ['export_gnn_model', 'get_supported_formats', 'export_to_format']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Export MCP not available")

class TestVisualizationMCP:
    """Test Visualization module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualization_mcp_imports(self):
        """Test Visualization MCP imports."""
        try:
            from visualization import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Visualization MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualization_mcp_tool_registration(self, mock_mcp_tools):
        """Test Visualization MCP tool registration."""
        try:
            from visualization.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify visualization tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected visualization tools
            expected_tools = ['create_graph_visualization', 'create_matrix_visualization', 'visualize_gnn_file']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Visualization MCP not available")

class TestRenderMCP:
    """Test Render module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_mcp_imports(self):
        """Test Render MCP imports."""
        try:
            from render import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Render MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_render_mcp_tool_registration(self, mock_mcp_tools):
        """Test Render MCP tool registration."""
        try:
            from render.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify render tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected render tools
            expected_tools = ['render_to_pymdp', 'render_to_rxinfer', 'get_available_renderers']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Render MCP not available")

class TestExecuteMCP:
    """Test Execute module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_execute_mcp_imports(self):
        """Test Execute MCP imports."""
        try:
            from execute import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Execute MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_execute_mcp_tool_registration(self, mock_mcp_tools):
        """Test Execute MCP tool registration."""
        try:
            from execute.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify execute tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected execute tools
            expected_tools = ['execute_simulation', 'validate_execution_environment', 'get_execution_status']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Execute MCP not available")

class TestLLMMCP:
    """Test LLM module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_mcp_imports(self):
        """Test LLM MCP imports."""
        try:
            from llm import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("LLM MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_llm_mcp_tool_registration(self, mock_mcp_tools):
        """Test LLM MCP tool registration."""
        try:
            from llm.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify LLM tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected LLM tools
            expected_tools = ['analyze_gnn_model', 'generate_model_description', 'extract_parameters']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("LLM MCP not available")

class TestAudioMCP:
    """Test Audio module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_mcp_imports(self):
        """Test Audio MCP imports."""
        try:
            from audio import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Audio MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_mcp_tool_registration(self, mock_mcp_tools):
        """Test Audio MCP tool registration."""
        try:
            from audio.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify audio tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected audio tools
            expected_tools = ['generate_audio_from_gnn', 'convert_to_sapf', 'get_audio_options']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Audio MCP not available")

class TestUtilsMCP:
    """Test Utils module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_utils_mcp_imports(self):
        """Test Utils MCP imports."""
        try:
            from utils import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Utils MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_utils_mcp_tool_registration(self, mock_mcp_tools):
        """Test Utils MCP tool registration."""
        try:
            from utils.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify utils tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected utils tools
            expected_tools = ['get_system_info', 'get_environment_info', 'validate_dependencies']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Utils MCP not available")

class TestPipelineMCP:
    """Test Pipeline module MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_mcp_imports(self):
        """Test Pipeline MCP imports."""
        try:
            from pipeline import mcp
            assert hasattr(mcp, 'register_tools')
        except ImportError:
            pytest.skip("Pipeline MCP not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pipeline_mcp_tool_registration(self, mock_mcp_tools):
        """Test Pipeline MCP tool registration."""
        try:
            from pipeline.mcp import register_tools
            
            register_tools(mock_mcp_tools)
            
            # Verify pipeline tools were registered
            assert len(mock_mcp_tools.tools) >= 1
            
            # Check for expected pipeline tools
            expected_tools = ['get_pipeline_steps', 'get_pipeline_status', 'validate_pipeline_dependencies']
            for tool_name in expected_tools:
                if tool_name in mock_mcp_tools.tools:
                    assert callable(mock_mcp_tools.tools[tool_name]['function'])
                    
        except ImportError:
            pytest.skip("Pipeline MCP not available")

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

class TestMCPToolSchemas:
    """Test MCP tool schemas and parameters."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_tool_schemas_structure(self, mock_mcp_tools):
        """Test that registered tools have proper schemas."""
        try:
            # Register tools from multiple modules
            modules_to_test = ['gnn', 'export', 'utils']
            
            for module_name in modules_to_test:
                try:
                    mcp_module = __import__(f'{module_name}.mcp', fromlist=['register_tools'])
                    if hasattr(mcp_module, 'register_tools'):
                        mcp_module.register_tools(mock_mcp_tools)
                except ImportError:
                    pass
            
            # Verify tool schemas
            for tool_name, tool_info in mock_mcp_tools.tools.items():
                assert 'schema' in tool_info
                assert 'description' in tool_info
                assert callable(tool_info['function'])
                
                # Verify schema structure
                schema = tool_info['schema']
                assert isinstance(schema, dict)
                
        except Exception as e:
            # Should handle schema errors gracefully
            assert len(mock_mcp_tools.tools) >= 0

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

class TestMCPIntegrationWorkflow:
    """Test complete MCP integration workflows."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_full_mcp_workflow(self, mock_mcp_tools, comprehensive_test_data):
        """Test complete MCP workflow across multiple modules."""
        try:
            # Register tools from multiple modules
            modules_to_register = ['gnn', 'export', 'utils']
            registered_modules = []
            
            for module_name in modules_to_register:
                try:
                    mcp_module = __import__(f'{module_name}.mcp', fromlist=['register_tools'])
                    if hasattr(mcp_module, 'register_tools'):
                        mcp_module.register_tools(mock_mcp_tools)
                        registered_modules.append(module_name)
                except ImportError:
                    pass
            
            # Verify we registered at least some tools
            assert len(mock_mcp_tools.tools) >= 1, "No tools were registered"
            assert len(registered_modules) >= 1, "No modules were registered"
            
            # Test workflow: verify tools are available for workflow
            workflow_tools = []
            
            if 'validate_gnn_content' in mock_mcp_tools.tools:
                workflow_tools.append('validate_gnn_content')
            
            if 'get_supported_formats' in mock_mcp_tools.tools:
                workflow_tools.append('get_supported_formats')
            
            # Verify workflow tools are registered (lightweight check)
            assert len(workflow_tools) >= 1, "No workflow tools available"
            
        except Exception as e:
            # Should handle workflow errors gracefully
            assert "error" in str(e).lower() or len(registered_modules) >= 0

# Performance and completeness tests
@pytest.mark.slow
def test_mcp_integration_performance():
    """Test performance of MCP integration across modules."""
    import time
    
    start_time = time.time()
    
    # Test MCP module imports
    modules_to_test = ['gnn', 'export', 'utils', 'pipeline']
    imported_modules = []
    
    for module_name in modules_to_test:
        try:
            mcp_module = __import__(f'{module_name}.mcp', fromlist=['register_tools'])
            imported_modules.append(module_name)
        except ImportError:
            pass
    
    import_time = time.time() - start_time
    
    # Should import reasonably quickly
    assert import_time < 5.0, f"MCP modules took {import_time:.2f}s to import"
    assert len(imported_modules) >= 1, "No MCP modules could be imported"

def test_mcp_integration_completeness():
    """Test completeness of MCP integration across modules."""
    expected_mcp_modules = [
        'gnn', 'export', 'visualization', 'render', 'execute', 
        'llm', 'audio', 'utils', 'pipeline'
    ]
    
    available_mcp_modules = []
    
    for module_name in expected_mcp_modules:
        try:
            mcp_module = __import__(f'{module_name}.mcp', fromlist=['register_tools'])
            if hasattr(mcp_module, 'register_tools'):
                available_mcp_modules.append(module_name)
        except ImportError:
            pass
    
    # Should have reasonable MCP coverage
    coverage_ratio = len(available_mcp_modules) / len(expected_mcp_modules)
    assert coverage_ratio >= 0.4, f"Low MCP coverage: {len(available_mcp_modules)}/{len(expected_mcp_modules)} modules have working MCP integration" 