"""
Comprehensive API Test Suite

This test suite covers all exposed functions, classes, and MCP integration
from all modules in the GNN pipeline system.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import all modules to test their exposed APIs
import src.gnn
import src.type_checker
import src.export
import src.visualization
import src.render
import src.execute
import src.llm
import src.sapf
import src.ontology
import src.mcp
import src.setup
import src.utils
import src.pipeline
import src.website


class TestGNNModule:
    """Test the GNN module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.gnn, 'validate_gnn_file')
        assert hasattr(src.gnn, 'GNNValidator')
        assert hasattr(src.gnn, 'GNNParser')
        assert hasattr(src.gnn, 'ValidationResult')
        assert hasattr(src.gnn, 'ParsedGNN')
        assert hasattr(src.gnn, 'get_module_info')
        assert hasattr(src.gnn, 'validate_gnn')
        assert hasattr(src.gnn, 'FEATURES')
        assert hasattr(src.gnn, '__version__')
    
    def test_get_module_info(self):
        """Test the get_module_info function."""
        info = src.gnn.get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'features' in info
        assert 'available_validators' in info
        assert 'available_parsers' in info
        assert 'schema_formats' in info
    
    def test_validate_gnn_function(self):
        """Test the validate_gnn function."""
        # Test with invalid input
        result = src.gnn.validate_gnn("invalid content")
        assert hasattr(result, 'is_valid') or isinstance(result, dict)
    
    def test_feature_flags(self):
        """Test that feature flags are properly set."""
        assert isinstance(src.gnn.FEATURES, dict)
        assert 'core_validation' in src.gnn.FEATURES
        assert src.gnn.FEATURES['core_validation'] is True


class TestExportModule:
    """Test the export module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.export, '_gnn_model_to_dict')
        assert hasattr(src.export, 'export_to_json_gnn')
        assert hasattr(src.export, 'export_to_xml_gnn')
        assert hasattr(src.export, 'export_to_python_pickle')
        assert hasattr(src.export, 'export_to_plaintext_summary')
        assert hasattr(src.export, 'export_to_plaintext_dsl')
        assert hasattr(src.export, 'get_module_info')
        assert hasattr(src.export, 'export_gnn_model')
        assert hasattr(src.export, 'get_supported_formats')
        assert hasattr(src.export, 'FEATURES')
        assert hasattr(src.export, '__version__')
    
    def test_get_module_info(self):
        """Test the get_module_info function."""
        info = src.export.get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'features' in info
        assert 'available_formats' in info
        assert 'graph_formats' in info
        assert 'text_formats' in info
        assert 'data_formats' in info
    
    def test_get_supported_formats(self):
        """Test the get_supported_formats function."""
        formats = src.export.get_supported_formats()
        assert isinstance(formats, dict)
        assert 'data_formats' in formats
        assert 'text_formats' in formats
        if src.export.HAS_NETWORKX:
            assert 'graph_formats' in formats
    
    def test_export_gnn_model_invalid_format(self):
        """Test export_gnn_model with invalid format."""
        result = src.export.export_gnn_model("nonexistent.gnn", "invalid_format")
        assert result["success"] is False
        assert "error" in result


class TestRenderModule:
    """Test the render module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.render, 'render_gnn_spec')
        assert hasattr(src.render, 'main')
        assert hasattr(src.render, 'get_module_info')
        assert hasattr(src.render, 'get_available_renderers')
        assert hasattr(src.render, 'FEATURES')
        assert hasattr(src.render, '__version__')
    
    def test_get_module_info(self):
        """Test the get_module_info function."""
        info = src.render.get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'features' in info
        assert 'available_targets' in info
        assert 'supported_formats' in info
    
    def test_get_available_renderers(self):
        """Test the get_available_renderers function."""
        renderers = src.render.get_available_renderers()
        assert isinstance(renderers, dict)
        
        # Check that available renderers have the expected structure
        for renderer_name, renderer_info in renderers.items():
            assert 'function' in renderer_info
            assert 'description' in renderer_info
            assert 'output_format' in renderer_info
    
    def test_feature_flags(self):
        """Test that feature flags are properly set."""
        assert isinstance(src.render.FEATURES, dict)
        assert 'pymdp_rendering' in src.render.FEATURES
        assert 'rxinfer_rendering' in src.render.FEATURES
        assert 'discopy_rendering' in src.render.FEATURES
        assert 'activeinference_jl_rendering' in src.render.FEATURES
        assert 'jax_rendering' in src.render.FEATURES
        assert 'mcp_integration' in src.render.FEATURES


class TestWebsiteModule:
    """Test the website module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.website, 'generate_html_report')
        assert hasattr(src.website, 'main_website_generator')
        assert hasattr(src.website, 'embed_image')
        assert hasattr(src.website, 'embed_markdown_file')
        assert hasattr(src.website, 'embed_text_file')
        assert hasattr(src.website, 'embed_json_file')
        assert hasattr(src.website, 'embed_html_file')
        assert hasattr(src.website, 'get_module_info')
        assert hasattr(src.website, 'get_supported_file_types')
        assert hasattr(src.website, 'FEATURES')
        assert hasattr(src.website, '__version__')
    
    def test_get_module_info(self):
        """Test the get_module_info function."""
        info = src.website.get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'features' in info
        assert 'supported_formats' in info
        assert 'embedding_capabilities' in info
    
    def test_get_supported_file_types(self):
        """Test the get_supported_file_types function."""
        file_types = src.website.get_supported_file_types()
        assert isinstance(file_types, dict)
        assert 'images' in file_types
        assert 'markdown' in file_types
        assert 'json' in file_types
        assert 'text' in file_types
        assert 'html' in file_types
    
    def test_generate_website_from_pipeline_output_nonexistent(self):
        """Test generate_website_from_pipeline_output with nonexistent directory."""
        result = src.website.generate_website_from_pipeline_output("/nonexistent/directory")
        assert result["success"] is False
        assert "error" in result


class TestSAPFModule:
    """Test the SAPF module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.sapf, 'SAPFGNNProcessor')
        assert hasattr(src.sapf, 'convert_gnn_to_sapf')
        assert hasattr(src.sapf, 'generate_audio_from_sapf')
        assert hasattr(src.sapf, 'validate_sapf_code')
        assert hasattr(src.sapf, 'SyntheticAudioGenerator')
        assert hasattr(src.sapf, 'generate_oscillator_audio')
        assert hasattr(src.sapf, 'apply_envelope')
        assert hasattr(src.sapf, 'mix_audio_channels')
        assert hasattr(src.sapf, 'get_module_info')
        assert hasattr(src.sapf, 'process_gnn_to_audio')
        assert hasattr(src.sapf, 'get_audio_generation_options')
        assert hasattr(src.sapf, 'FEATURES')
        assert hasattr(src.sapf, '__version__')
    
    def test_get_module_info(self):
        """Test the get_module_info function."""
        info = src.sapf.get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'features' in info
        assert 'audio_capabilities' in info
        assert 'supported_formats' in info
    
    def test_get_audio_generation_options(self):
        """Test the get_audio_generation_options function."""
        options = src.sapf.get_audio_generation_options()
        assert isinstance(options, dict)
        assert 'oscillators' in options
        assert 'envelopes' in options
        assert 'effects' in options
        assert 'output_formats' in options
    
    def test_process_gnn_to_audio_invalid_input(self):
        """Test process_gnn_to_audio with invalid input."""
        result = src.sapf.process_gnn_to_audio("", "test_model", "/tmp")
        # This should fail due to empty GNN content
        assert result["success"] is False or "error" in result


class TestOntologyModule:
    """Test the ontology module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.ontology, 'parse_gnn_ontology_section')
        assert hasattr(src.ontology, 'load_defined_ontology_terms')
        assert hasattr(src.ontology, 'validate_annotations')
        assert hasattr(src.ontology, 'generate_ontology_report_for_file')
        assert hasattr(src.ontology, 'get_mcp_interface')
        assert hasattr(src.ontology, 'get_module_info')
        assert hasattr(src.ontology, 'process_gnn_ontology')
        assert hasattr(src.ontology, 'get_ontology_processing_options')
        assert hasattr(src.ontology, 'FEATURES')
        assert hasattr(src.ontology, '__version__')
    
    def test_get_module_info(self):
        """Test the get_module_info function."""
        info = src.ontology.get_module_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'description' in info
        assert 'features' in info
        assert 'processing_capabilities' in info
        assert 'supported_formats' in info
    
    def test_get_ontology_processing_options(self):
        """Test the get_ontology_processing_options function."""
        options = src.ontology.get_ontology_processing_options()
        assert isinstance(options, dict)
        assert 'parsing_options' in options
        assert 'validation_options' in options
        assert 'report_formats' in options
        assert 'output_options' in options
    
    def test_process_gnn_ontology_nonexistent_file(self):
        """Test process_gnn_ontology with nonexistent file."""
        result = src.ontology.process_gnn_ontology("/nonexistent/file.gnn")
        assert result["success"] is False
        assert "error" in result
    
    def test_parse_gnn_ontology_section_empty(self):
        """Test parse_gnn_ontology_section with empty content."""
        result = src.ontology.parse_gnn_ontology_section("")
        assert isinstance(result, dict)
        assert len(result) == 0


class TestTypeCheckerModule:
    """Test the type checker module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        # Import the main checker module
        from src.type_checker import checker
        assert hasattr(checker, 'GNNTypeChecker')
        assert hasattr(checker, 'TypeCheckResult')
        assert hasattr(checker, 'check_gnn_file')
        assert hasattr(checker, 'validate_syntax')
        assert hasattr(checker, 'estimate_resources')
    
    def test_type_checker_instantiation(self):
        """Test that the type checker can be instantiated."""
        from src.type_checker.checker import GNNTypeChecker
        checker = GNNTypeChecker()
        assert checker is not None


class TestVisualizationModule:
    """Test the visualization module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        # Import the main visualization module
        from src.visualization import visualizer
        assert hasattr(visualizer, 'GNNVisualizer')
        assert hasattr(visualizer, 'generate_graph_visualization')
        assert hasattr(visualizer, 'generate_matrix_visualization')
        assert hasattr(visualizer, 'create_visualization_report')
    
    def test_visualizer_instantiation(self):
        """Test that the visualizer can be instantiated."""
        from src.visualization.visualizer import GNNVisualizer
        visualizer = GNNVisualizer()
        assert visualizer is not None


class TestExecuteModule:
    """Test the execute module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        # Import the main execute module
        from src.execute import executor
        assert hasattr(executor, 'GNNExecutor')
        assert hasattr(executor, 'execute_gnn_model')
        assert hasattr(executor, 'run_simulation')
        assert hasattr(executor, 'generate_execution_report')
    
    def test_executor_instantiation(self):
        """Test that the executor can be instantiated."""
        from src.execute.executor import GNNExecutor
        executor = GNNExecutor()
        assert executor is not None


class TestLLMModule:
    """Test the LLM module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        # Import the main LLM module
        from src.llm import llm_processor
        assert hasattr(llm_processor, 'GNNLLMProcessor')
        assert hasattr(llm_processor, 'analyze_gnn_model')
        assert hasattr(llm_processor, 'generate_explanation')
        assert hasattr(llm_processor, 'enhance_model')
    
    def test_llm_processor_instantiation(self):
        """Test that the LLM processor can be instantiated."""
        from src.llm.llm_processor import GNNLLMProcessor
        processor = GNNLLMProcessor()
        assert processor is not None


class TestMCPModule:
    """Test the MCP module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.mcp, 'MCPServer')
        assert hasattr(src.mcp, 'register_module_tools')
        assert hasattr(src.mcp, 'start_mcp_server')
        assert hasattr(src.mcp, 'get_available_tools')
    
    def test_mcp_server_instantiation(self):
        """Test that the MCP server can be instantiated."""
        from src.mcp.server import MCPServer
        server = MCPServer()
        assert server is not None


class TestSetupModule:
    """Test the setup module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.setup, 'setup_environment')
        assert hasattr(src.setup, 'install_dependencies')
        assert hasattr(src.setup, 'validate_system')
        assert hasattr(src.setup, 'get_environment_info')
    
    def test_setup_functions_exist(self):
        """Test that setup functions exist and are callable."""
        assert callable(src.setup.setup_environment)
        assert callable(src.setup.install_dependencies)
        assert callable(src.setup.validate_system)
        assert callable(src.setup.get_environment_info)


class TestUtilsModule:
    """Test the utils module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.utils, 'EnhancedArgumentParser')
        assert hasattr(src.utils, 'PipelineLogger')
        assert hasattr(src.utils, 'performance_tracker')
        assert hasattr(src.utils, 'validate_pipeline_dependencies')
        assert hasattr(src.utils, 'setup_step_logging')
    
    def test_utils_classes_instantiation(self):
        """Test that utility classes can be instantiated."""
        from src.utils.argument_utils import EnhancedArgumentParser
        parser = EnhancedArgumentParser()
        assert parser is not None


class TestPipelineModule:
    """Test the pipeline module's exposed API."""
    
    def test_module_imports(self):
        """Test that all expected functions are available."""
        assert hasattr(src.pipeline, 'STEP_METADATA')
        assert hasattr(src.pipeline, 'get_pipeline_config')
        assert hasattr(src.pipeline, 'get_output_dir_for_script')
        assert hasattr(src.pipeline, 'execute_pipeline_step')
    
    def test_pipeline_config(self):
        """Test that pipeline configuration is accessible."""
        assert isinstance(src.pipeline.STEP_METADATA, dict)
        assert len(src.pipeline.STEP_METADATA) > 0


class TestMCPIntegration:
    """Test MCP integration across all modules."""
    
    def test_mcp_availability_flags(self):
        """Test that MCP availability flags are properly set."""
        modules_with_mcp = [
            src.gnn, src.export, src.render, src.website, 
            src.sapf, src.ontology, src.mcp, src.setup
        ]
        
        for module in modules_with_mcp:
            if hasattr(module, 'MCP_AVAILABLE'):
                assert isinstance(module.MCP_AVAILABLE, bool)
            if hasattr(module, 'FEATURES'):
                assert 'mcp_integration' in module.FEATURES
    
    def test_register_tools_functions(self):
        """Test that register_tools functions exist where expected."""
        modules_with_register_tools = [
            src.export, src.render, src.website, src.sapf, src.ontology
        ]
        
        for module in modules_with_register_tools:
            if hasattr(module, 'MCP_AVAILABLE') and module.MCP_AVAILABLE:
                assert hasattr(module, 'register_tools')
                assert callable(module.register_tools)


class TestModuleConsistency:
    """Test consistency across all modules."""
    
    def test_version_consistency(self):
        """Test that all modules have version information."""
        modules = [
            src.gnn, src.export, src.render, src.website, 
            src.sapf, src.ontology, src.mcp, src.setup
        ]
        
        for module in modules:
            assert hasattr(module, '__version__')
            assert isinstance(module.__version__, str)
            assert len(module.__version__) > 0
    
    def test_feature_flags_consistency(self):
        """Test that all modules have consistent feature flag structure."""
        modules_with_features = [
            src.gnn, src.export, src.render, src.website, 
            src.sapf, src.ontology
        ]
        
        for module in modules_with_features:
            assert hasattr(module, 'FEATURES')
            assert isinstance(module.FEATURES, dict)
            assert len(module.FEATURES) > 0
    
    def test_module_info_consistency(self):
        """Test that all modules have consistent get_module_info structure."""
        modules_with_info = [
            src.gnn, src.export, src.render, src.website, 
            src.sapf, src.ontology
        ]
        
        for module in modules_with_info:
            assert hasattr(module, 'get_module_info')
            assert callable(module.get_module_info)
            
            info = module.get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'features' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 