"""
Test Categories - Modular Test Category Definitions

This module defines the test category configuration for modular test execution.
Each category has its own timeout, max failures, parallel execution settings,
and list of test files.
"""

from typing import Dict, Any, List

# Test category definitions for modular test execution
MODULAR_TEST_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "gnn": {
        "name": "GNN Module Tests",
        "description": "GNN processing and validation tests",
        "files": ["test_gnn_overall.py", "test_gnn_parsing.py", "test_gnn_validation.py", 
                  "test_gnn_processing.py", "test_gnn_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "render": {
        "name": "Render Module Tests", 
        "description": "Code generation and rendering tests",
        "files": ["test_render_overall.py", "test_render_integration.py", "test_render_performance.py",
                  "test_activeinference_matrix_formatting.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "mcp": {
        "name": "MCP Module Tests",
        "description": "Model Context Protocol tests",
        "files": ["test_mcp_overall.py", "test_mcp_tools.py", "test_mcp_transport.py", 
                  "test_mcp_integration.py", "test_mcp_performance.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "audio": {
        "name": "Audio Module Tests",
        "description": "Audio generation and SAPF tests",
        "files": ["test_audio_overall.py", "test_audio_sapf.py", "test_audio_generation.py", 
                  "test_audio_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "visualization": {
        "name": "Visualization Module Tests",
        "description": "Graph and matrix visualization tests",
        "files": ["test_visualization_overall.py", "test_visualization_matrices.py", 
                  "test_visualization_ontology.py", "test_visualization_comprehensive.py",
                  "test_d2_visualizer.py", "test_mermaid_converter.py", "test_mermaid_parser.py",
                  "test_oxdraw_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "pipeline": {
        "name": "Pipeline Module Tests",
        "description": "Pipeline orchestration and step tests",
        "files": ["test_pipeline_overall.py", "test_pipeline_integration.py", 
                  "test_pipeline_orchestration.py", "test_pipeline_performance.py", 
                  "test_pipeline_recovery.py", "test_pipeline_scripts.py", 
                  "test_pipeline_infrastructure.py", "test_pipeline_functionality.py",
                  "test_pipeline_render_execute_analyze.py", "test_pipeline_error_scenarios.py",
                  "test_pipeline_improvements_validation.py", "test_pipeline_main.py",
                  "test_pipeline_warnings_fix.py", "test_error_recovery_framework.py",
                  "test_infrastructure_consistency.py"],
        "markers": [],
        "timeout_seconds": 1800,
        "max_failures": 10,
        "parallel": False
    },
    "export": {
        "name": "Export Module Tests",
        "description": "Multi-format export tests",
        "files": ["test_export_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "execute": {
        "name": "Execute Module Tests",
        "description": "Execution and simulation tests including PyMDP",
        "files": [
            "test_execute_overall.py",
            "test_execute_path_collection.py",
            "test_execute_pymdp_integration.py",
            "test_execute_pymdp_integration_module.py",
            "test_execute_pymdp_package.py",
            "test_execute_pymdp_simulation.py",
            "test_execute_pymdp_utils.py",
            "test_execute_pymdp_visualization_module.py",
            "test_execute_pymdp_visualizer.py",
        ],
        "markers": [],
        "timeout_seconds": 300,
        "max_failures": 10,
        "parallel": True
    },
    "llm": {
        "name": "LLM Module Tests",
        "description": "LLM integration and analysis tests",
        "files": ["test_llm_overall.py", "test_llm_ollama.py", "test_llm_ollama_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "ontology": {
        "name": "Ontology Module Tests",
        "description": "Ontology processing and validation tests",
        "files": ["test_ontology_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "website": {
        "name": "Website Module Tests",
        "description": "Website generation tests",
        "files": ["test_website_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "report": {
        "name": "Report Module Tests",
        "description": "Report generation and formatting tests",
        "files": ["test_report_overall.py", "test_report_generation.py", 
                  "test_report_integration.py", "test_report_formats.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "environment": {
        "name": "Environment Module Tests",
        "description": "Environment setup and validation tests",
        "files": ["test_environment_overall.py", "test_environment_dependencies.py",
                  "test_environment_integration.py", "test_environment_python.py",
                  "test_environment_system.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "type_checker": {
        "name": "Type Checker Module Tests",
        "description": "Type checking and validation tests",
        "files": ["test_type_checker_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "validation": {
        "name": "Validation Module Tests",
        "description": "Validation and consistency tests",
        "files": ["test_validation_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "model_registry": {
        "name": "Model Registry Module Tests",
        "description": "Model registry and versioning tests",
        "files": ["test_model_registry_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "analysis": {
        "name": "Analysis Module Tests",
        "description": "Analysis and statistical tests",
        "files": ["test_analysis_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "integration": {
        "name": "Integration Module Tests",
        "description": "System integration tests",
        "files": ["test_integration_overall.py", "test_integration_processor.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "security": {
        "name": "Security Module Tests",
        "description": "Security validation tests",
        "files": ["test_security_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "research": {
        "name": "Research Module Tests",
        "description": "Research tools tests",
        "files": ["test_research_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "ml_integration": {
        "name": "ML Integration Module Tests",
        "description": "Machine learning integration tests",
        "files": ["test_ml_integration_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "advanced_visualization": {
        "name": "Advanced Visualization Module Tests",
        "description": "Advanced visualization tests",
        "files": ["test_advanced_visualization_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "gui": {
        "name": "GUI Module Tests",
        "description": "GUI functionality and widget tests",
        "files": ["test_gui_overall.py", "test_gui_functionality.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "comprehensive": {
        "name": "Comprehensive API Tests",
        "description": "Comprehensive API and integration tests",
        "files": ["test_comprehensive_api.py", "test_core_modules.py", "test_fast_suite.py",
                  "test_main_orchestrator.py", "test_coverage_overall.py", "test_performance_overall.py",
                  "test_unit_overall.py", "test_coverage_assessment.py", "test_new_utils.py",
                  "test_performance_baselines.py", "test_runner_helper.py"],
        "markers": [],
        "timeout_seconds": 300,
        "max_failures": 15,
        "parallel": False
    }
}


def get_category_names() -> List[str]:
    """Get list of all category names."""
    return list(MODULAR_TEST_CATEGORIES.keys())


def get_category(name: str) -> Dict[str, Any]:
    """Get category configuration by name."""
    return MODULAR_TEST_CATEGORIES.get(name, {})


def get_category_files(name: str) -> List[str]:
    """Get test files for a specific category."""
    category = MODULAR_TEST_CATEGORIES.get(name, {})
    return category.get("files", [])


def get_all_test_files() -> List[str]:
    """Get all test files across all categories."""
    files = []
    for category in MODULAR_TEST_CATEGORIES.values():
        files.extend(category.get("files", []))
    return list(set(files))  # Deduplicate


__all__ = [
    "MODULAR_TEST_CATEGORIES",
    "get_category_names",
    "get_category",
    "get_category_files", 
    "get_all_test_files",
]
