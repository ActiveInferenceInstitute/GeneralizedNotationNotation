"""
Comprehensive API Test Suite

This test suite covers all exposed functions, classes, and MCP integration
from all modules in the GNN pipeline system.
"""

from pathlib import Path
from typing import Any

import pytest

# Test markers
pytestmark: list[Any] = [pytest.mark.integration]

# Import all modules to test their exposed APIs - with error handling
try:
    import gnn

    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

try:
    import type_checker

    TYPE_CHECKER_AVAILABLE = True
except ImportError:
    TYPE_CHECKER_AVAILABLE = False

try:
    import export

    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

try:
    import visualization

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import render

    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False

try:
    import execute

    EXECUTE_AVAILABLE = True
except ImportError:
    EXECUTE_AVAILABLE = False

try:
    import llm

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    import audio

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import analysis

    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

try:
    import integration

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    import security

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

try:
    import research

    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False

try:
    import website

    WEBSITE_AVAILABLE = True
except ImportError:
    WEBSITE_AVAILABLE = False

try:
    import report

    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

try:
    import ontology

    ONTOLOGY_AVAILABLE = True
except ImportError:
    ONTOLOGY_AVAILABLE = False

try:
    import mcp

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    import setup

    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

try:
    import utils

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

try:
    import pipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


class TestGNNModule:
    """Test the GNN module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(gnn, "validate_gnn_file") or hasattr(gnn, "process_gnn")
        # Test for any available functions without being too strict
        attrs = dir(gnn)
        assert len([attr for attr in attrs if not attr.startswith("_")]) > 0

    def test_get_module_info(self) -> None:
        """Test the get_module_info function."""
        info = gnn.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert "available_validators" in info
        assert "available_parsers" in info
        assert "schema_formats" in info

    def test_validate_gnn_function(self) -> None:
        """Test the validate_gnn function."""
        # Test with invalid input — validate_gnn returns tuple(bool, list[str])
        result = gnn.validate_gnn("invalid content")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        is_valid, messages = result
        assert isinstance(is_valid, bool)
        assert isinstance(messages, list)

    def test_feature_flags(self) -> None:
        """Test that feature flags are properly set."""
        assert isinstance(gnn.FEATURES, dict)
        assert "core_validation" in gnn.FEATURES
        assert gnn.FEATURES["core_validation"] is True


class TestExportModule:
    """Test the export module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(export, "_gnn_model_to_dict")
        assert hasattr(export, "export_to_json_gnn")
        assert hasattr(export, "export_to_xml_gnn")
        assert hasattr(export, "export_to_python_pickle")
        assert hasattr(export, "export_to_plaintext_summary")
        assert hasattr(export, "export_to_plaintext_dsl")
        assert hasattr(export, "get_module_info")
        assert hasattr(export, "export_gnn_model")
        assert hasattr(export, "get_supported_formats")
        assert hasattr(export, "FEATURES")
        assert hasattr(export, "__version__")

    def test_get_module_info(self) -> None:
        """Test the get_module_info function."""
        info = export.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert "available_formats" in info
        assert "graph_formats" in info
        assert "text_formats" in info
        assert "data_formats" in info

    def test_get_supported_formats(self) -> None:
        """Test the get_supported_formats function."""
        formats = export.get_supported_formats_dict()
        assert isinstance(formats, dict)
        assert "data_formats" in formats
        assert "text_formats" in formats
        if export.HAS_NETWORKX:
            assert "graph_formats" in formats

    def test_export_gnn_model_invalid_format(self, tmp_path: Path) -> None:
        """Test export_gnn_model with invalid format."""
        result = export.export_gnn_model({}, tmp_path, formats=["invalid_format"])
        assert result["success"] is False
        assert "error" in result


class TestRenderModule:
    """Test the render module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(render, "render_gnn_spec")
        assert hasattr(render, "main")
        assert hasattr(render, "get_module_info")
        assert hasattr(render, "get_available_renderers")
        assert hasattr(render, "FEATURES")
        assert hasattr(render, "__version__")

    def test_get_module_info(self) -> None:
        """Test the get_module_info function."""
        info = render.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert "available_targets" in info
        assert "supported_formats" in info

    def test_get_available_renderers(self) -> None:
        """Test the get_available_renderers function."""
        renderers = render.get_available_renderers()
        assert isinstance(renderers, dict)

        # Check that available renderers have the expected structure
        for _renderer_name, renderer_info in renderers.items():
            assert "function" in renderer_info
            assert "description" in renderer_info
            assert "output_format" in renderer_info

    def test_feature_flags(self) -> None:
        """Test that feature flags are properly set."""
        assert isinstance(render.FEATURES, dict)
        assert "pymdp_rendering" in render.FEATURES
        assert "rxinfer_rendering" in render.FEATURES
        assert "discopy_rendering" in render.FEATURES
        assert "activeinference_jl_rendering" in render.FEATURES
        assert "jax_rendering" in render.FEATURES
        assert "mcp_integration" in render.FEATURES


class TestWebsiteModule:
    """Test the website module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(website, "generate_website")
        assert hasattr(website, "generate_html_report")
        assert hasattr(website, "embed_image")
        assert hasattr(website, "embed_markdown_file")
        assert hasattr(website, "embed_text_file")
        assert hasattr(website, "embed_json_file")
        assert hasattr(website, "embed_html_file")
        assert hasattr(website, "get_module_info")
        assert hasattr(website, "get_supported_file_types")
        assert hasattr(website, "FEATURES")
        assert hasattr(website, "__version__")

    def test_get_module_info(self) -> None:
        """Test the get_module_info function."""
        info = website.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert "supported_file_types" in info
        assert "embedding_capabilities" in info

    def test_get_supported_file_types(self) -> None:
        """Test the get_supported_file_types function."""
        file_types = website.SUPPORTED_FILE_TYPES
        assert isinstance(file_types, dict)
        assert "images" in file_types
        assert "markdown" in file_types
        assert "json" in file_types
        assert "text" in file_types
        assert "html" in file_types

    def test_generate_website_from_pipeline_output_nonexistent(self) -> None:
        """Test generate_website with nonexistent directory."""
        import logging
        from pathlib import Path

        # Test with nonexistent directory
        logger = logging.getLogger("test")
        nonexistent_dir = Path("/nonexistent/directory")
        output_dir = Path("/tmp/test_output")  # nosec B108

        try:
            # This should handle the error gracefully
            result = website.generate_website(logger, nonexistent_dir, output_dir)
            # If it returns a result, it should indicate failure
            if isinstance(result, dict):
                assert result.get("success") is False
        except Exception as e:
            # Expected to fail due to nonexistent directory
            assert "nonexistent" in str(e).lower() or "not found" in str(e).lower()

    def test_validate_website_config(self) -> None:
        """Test website config validation."""
        # Test with valid config (using existing directory)
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            config: dict[str, Any] = {"output_dir": temp_dir}
            result = website.validate_website_config(config)
            assert isinstance(result, dict)
            assert result["valid"]

        # Test with invalid config
        config = {"output_dir": "/nonexistent/directory"}
        result = website.validate_website_config(config)
        assert isinstance(result, dict)
        assert not result["valid"]
        assert len(result["errors"]) > 0


class TestSAPFModule:
    """Test the SAPF module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(audio, "SAPFGNNProcessor")
        assert hasattr(audio, "convert_gnn_to_sapf")
        assert hasattr(audio, "generate_audio_from_sapf")
        assert hasattr(audio, "validate_sapf_code")
        assert hasattr(audio, "SyntheticAudioGenerator")
        assert hasattr(audio, "generate_oscillator_audio")
        assert hasattr(audio, "apply_envelope")
        assert hasattr(audio, "mix_audio_channels")
        assert hasattr(audio, "get_module_info")
        assert hasattr(audio, "process_gnn_to_audio")
        assert hasattr(audio, "get_audio_generation_options")
        assert hasattr(audio, "FEATURES")
        assert hasattr(audio, "__version__")

    def test_get_module_info(self) -> None:
        """Test the get_module_info function."""
        info = audio.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert "audio_capabilities" in info
        assert "supported_formats" in info

    def test_get_audio_generation_options(self) -> None:
        """Test the get_audio_generation_options function."""
        options = audio.get_audio_generation_options()
        assert isinstance(options, dict)
        assert "oscillators" in options
        assert "envelopes" in options
        assert "effects" in options
        assert "output_formats" in options

    def test_process_gnn_to_audio_invalid_input(self) -> None:
        """Test process_gnn_to_audio with invalid input."""
        result = audio.process_gnn_to_audio("", "test_model", "/tmp")  # nosec B108
        # This should fail due to empty GNN content
        assert result["success"] is False or "error" in result


class TestOntologyModule:
    """Test the ontology module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(ontology, "parse_gnn_ontology_section")
        assert hasattr(ontology, "load_defined_ontology_terms")
        assert hasattr(ontology, "validate_annotations")
        assert hasattr(ontology, "generate_ontology_report_for_file")
        assert hasattr(ontology, "get_mcp_interface")
        assert hasattr(ontology, "get_module_info")
        assert hasattr(ontology, "process_gnn_ontology")
        assert hasattr(ontology, "get_ontology_processing_options")
        assert hasattr(ontology, "FEATURES")
        assert hasattr(ontology, "__version__")

    def test_get_module_info(self) -> None:
        """Test the get_module_info function."""
        info = ontology.get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "features" in info
        assert "processing_capabilities" in info
        assert "supported_formats" in info

    def test_get_ontology_processing_options(self) -> None:
        """Test the get_ontology_processing_options function."""
        options = ontology.get_ontology_processing_options()
        assert isinstance(options, dict)
        assert "parsing_options" in options
        assert "validation_options" in options
        assert "report_formats" in options
        assert "output_options" in options

    def test_process_gnn_ontology_nonexistent_file(self) -> None:
        """Test process_gnn_ontology with nonexistent file."""
        result = ontology.process_gnn_ontology("/nonexistent/file.gnn")
        assert result["success"] is False
        assert "error" in result

    def test_parse_gnn_ontology_section_empty(self) -> None:
        """Test parse_gnn_ontology_section with empty content."""
        result = ontology.parse_gnn_ontology_section("")
        assert isinstance(result, dict)
        assert len(result) == 0


class TestTypeCheckerModule:
    """Test the type checker module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        # Verify the public API surface exported by type_checker/__init__.py
        assert hasattr(type_checker, "GNNTypeChecker")
        assert hasattr(type_checker, "estimate_file_resources")
        assert hasattr(type_checker, "__version__")
        assert hasattr(type_checker, "FEATURES")

    def test_type_checker_instantiation(self) -> None:
        """Test that the type checker can be instantiated."""
        from type_checker.processor import GNNTypeChecker

        checker = GNNTypeChecker()
        assert checker is not None


class TestVisualizationModule:
    """Test the visualization module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        # Import the main visualization module
        from visualization import visualizer

        assert hasattr(visualizer, "GNNVisualizer")
        assert hasattr(visualizer, "generate_graph_visualization")
        assert hasattr(visualizer, "generate_matrix_visualization")
        assert hasattr(visualizer, "create_visualization_report")

    def test_visualizer_instantiation(self) -> None:
        """Test that the visualizer can be instantiated."""
        from visualization.visualizer import GNNVisualizer

        visualizer = GNNVisualizer()
        assert visualizer is not None


class TestExecuteModule:
    """Test the execute module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        # Import the main execute module
        from execute import executor

        assert hasattr(executor, "GNNExecutor")
        assert hasattr(executor, "execute_gnn_model")
        assert hasattr(executor, "run_simulation")
        assert hasattr(executor, "generate_execution_report")

    def test_executor_instantiation(self) -> None:
        """Test that the executor can be instantiated."""
        from execute.executor import GNNExecutor

        executor = GNNExecutor()
        assert executor is not None


class TestLLMModule:
    """Test the LLM module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        # Import the main LLM module
        from llm import llm_processor

        assert hasattr(llm_processor, "GNNLLMProcessor")
        assert hasattr(llm_processor, "analyze_gnn_model")
        assert hasattr(llm_processor, "generate_explanation")
        assert hasattr(llm_processor, "enhance_model")

    def test_llm_processor_instantiation(self) -> None:
        """Test that the LLM processor can be instantiated."""
        from llm.llm_processor import GNNLLMProcessor

        processor = GNNLLMProcessor()
        assert processor is not None


class TestMCPModule:
    """Test the MCP module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(mcp, "MCP")
        assert hasattr(mcp, "register_module_tools")
        assert hasattr(mcp, "initialize")
        assert hasattr(mcp, "get_available_tools")

    def test_mcp_server_instantiation(self) -> None:
        """Test that the MCP server can be instantiated."""
        from mcp.server import MCPServer

        server = MCPServer()
        assert server is not None


class TestSetupModule:
    """Test the setup module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(setup, "setup_environment")
        assert hasattr(setup, "install_dependencies")
        assert hasattr(setup, "validate_system")
        assert hasattr(setup, "get_environment_info")

    def test_setup_functions_exist(self) -> None:
        """Test that setup functions exist and are callable."""
        assert callable(setup.setup_environment)
        assert callable(setup.install_dependencies)
        assert callable(setup.validate_system)
        assert callable(setup.get_environment_info)


class TestUtilsModule:
    """Test the utils module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(utils, "ArgumentParser")
        assert hasattr(utils, "PipelineLogger")
        assert hasattr(utils, "performance_tracker")
        assert hasattr(utils, "validate_pipeline_dependencies")
        assert hasattr(utils, "setup_step_logging")

    def test_utils_classes_instantiation(self) -> None:
        """Test that utility classes can be instantiated."""
        from utils.argument_utils import ArgumentParser

        parser = ArgumentParser()
        assert parser is not None


class TestPipelineModule:
    """Test the pipeline module's exposed API."""

    def test_module_imports(self) -> None:
        """Test that all expected functions are available."""
        assert hasattr(pipeline, "STEP_METADATA")
        assert hasattr(pipeline, "get_pipeline_config")
        assert hasattr(pipeline, "get_output_dir_for_script")
        assert hasattr(pipeline, "execute_pipeline_step")

    def test_pipeline_config(self) -> None:
        """Test that pipeline configuration is accessible."""
        assert isinstance(pipeline.STEP_METADATA, dict)
        assert len(pipeline.STEP_METADATA) > 0


class TestMCPIntegration:
    """Test MCP integration across all modules."""

    def test_mcp_availability_flags(self) -> None:
        """Test that MCP availability flags are properly set."""
        modules_with_mcp: list[Any] = [
            gnn,
            export,
            render,
            website,
            audio,
            ontology,
            mcp,
        ]

        for module in modules_with_mcp:
            if hasattr(module, "MCP_AVAILABLE"):
                assert isinstance(module.MCP_AVAILABLE, bool)
            if hasattr(module, "FEATURES"):
                assert "mcp_integration" in module.FEATURES

    def test_register_tools_functions(self) -> None:
        """Test that register_tools functions exist where expected."""
        modules_with_register_tools: list[Any] = [
            export,
            render,
            website,
            audio,
            ontology,
        ]

        for module in modules_with_register_tools:
            if hasattr(module, "MCP_AVAILABLE") and module.MCP_AVAILABLE:
                assert hasattr(module, "register_tools")
                assert callable(module.register_tools)


class TestModuleConsistency:
    """Test consistency across all modules."""

    def test_version_consistency(self) -> None:
        """Test that all modules have version information."""
        modules: list[Any] = [
            gnn,
            export,
            render,
            website,
            audio,
            ontology,
            mcp,
        ]

        for module in modules:
            assert hasattr(module, "__version__")
            assert isinstance(module.__version__, str)
            assert len(module.__version__) > 0

    def test_feature_flags_consistency(self) -> None:
        """Test that all modules have consistent feature flag structure."""
        modules_with_features: list[Any] = [
            gnn,
            export,
            render,
            website,
            audio,
            ontology,
        ]

        for module in modules_with_features:
            assert hasattr(module, "FEATURES")
            assert isinstance(module.FEATURES, dict)
            assert len(module.FEATURES) > 0

    def test_module_info_consistency(self) -> None:
        """Test that all modules have consistent get_module_info structure."""
        modules_with_info: list[Any] = [
            gnn,
            export,
            render,
            website,
            audio,
            ontology,
        ]

        for module in modules_with_info:
            assert hasattr(module, "get_module_info")
            assert callable(module.get_module_info)

            info = module.get_module_info()
            assert isinstance(info, dict)
            assert "version" in info
            assert "description" in info
            assert "features" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
