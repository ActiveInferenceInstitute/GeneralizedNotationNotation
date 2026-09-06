"""
Test Categories - Modular Test Category Definitions

This module defines the test category configuration for modular test
execution. Each category has its own timeout, max failures, parallel
execution settings, and list of test files.

The category system is the routing table used by ``_ModularTestRunner``
(``test_runner_modular.py``): one entry per test group, each naming the
test files that belong to it (relative to ``src/tests/``). Missing files
are skipped by discovery; ``missing_category_files()`` reports them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypedDict


class TestCategory(TypedDict, total=False):
    """Configuration for one modular test category.

    All keys are optional at the type level for backward compatibility with
    partial dicts, but every shipped category defines all of them.
    """

    name: str
    description: str
    files: List[str]
    markers: List[str]
    timeout_seconds: int
    max_failures: int
    parallel: bool


# Test category definitions for modular test execution
MODULAR_TEST_CATEGORIES: Dict[str, TestCategory] = {
    "gnn": {
        "name": "GNN Module Tests",
        "description": "GNN processing and validation tests",
        "files": [
            "gnn/test_gnn_overall.py",
            "gnn/test_gnn_parsing.py",
            "gnn/test_gnn_processing.py",
            "gnn/test_gnn_validation.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "render": {
        "name": "Render Module Tests",
        "description": "Code generation and rendering tests",
        "files": [
            "render/test_activeinference_matrix_formatting.py",
            "render/test_render_integration.py",
            "render/test_render_overall.py",
            "render/test_render_performance.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "mcp": {
        "name": "MCP Module Tests",
        "description": "Model Context Protocol tests",
        "files": [
            "mcp/test_mcp_audit.py",
            "mcp/test_mcp_configurability.py",
            "mcp/test_mcp_functional.py",
            "mcp/test_mcp_overall.py",
            "mcp/test_mcp_performance.py",
            "mcp/test_mcp_tools.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "audio": {
        "name": "Audio Module Tests",
        "description": "Audio generation and SAPF tests",
        "files": [
            "audio/test_audio_generation.py",
            "audio/test_audio_integration.py",
            "audio/test_audio_overall.py",
            "audio/test_audio_sapf.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "visualization": {
        "name": "Visualization Module Tests",
        "description": "Graph and matrix visualization tests",
        "files": [
            "gui/test_oxdraw_integration.py",
            "visualization/test_d2_visualizer.py",
            "visualization/test_mermaid_converter.py",
            "visualization/test_mermaid_parser.py",
            "visualization/test_visualization_comprehensive.py",
            "visualization/test_visualization_matrices.py",
            "visualization/test_visualization_ontology.py",
            "visualization/test_visualization_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "pipeline": {
        "name": "Pipeline Module Tests",
        "description": "Pipeline orchestration and step tests",
        "files": [
            "pipeline/test_pipeline_error_scenarios.py",
            "pipeline/test_pipeline_functionality.py",
            "pipeline/test_pipeline_improvements_validation.py",
            "pipeline/test_pipeline_infrastructure.py",
            "pipeline/test_pipeline_integration.py",
            "pipeline/test_pipeline_main.py",
            "pipeline/test_pipeline_orchestration.py",
            "pipeline/test_pipeline_overall.py",
            "pipeline/test_pipeline_performance.py",
            "pipeline/test_pipeline_recovery.py",
            "pipeline/test_pipeline_render_execute_analyze.py",
            "pipeline/test_pipeline_scripts.py",
            "utils/test_error_recovery_framework.py",
            "utils/test_pipeline_warnings_fix.py",
        ],
        "markers": [],
        "timeout_seconds": 1800,
        "max_failures": 10,
        "parallel": False,
    },
    "export": {
        "name": "Export Module Tests",
        "description": "Multi-format export tests",
        "files": [
            "export/test_export_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "execute": {
        "name": "Execute Module Tests",
        "description": "Execution and simulation tests including PyMDP",
        "files": [
            "execute/test_execute_overall.py",
            "execute/test_execute_path_collection.py",
            "execute/test_execute_pymdp_integration.py",
            "execute/test_execute_pymdp_integration_module.py",
            "execute/test_execute_pymdp_package.py",
            "execute/test_execute_pymdp_simulation.py",
            "execute/test_execute_pymdp_utils.py",
            "execute/test_execute_pymdp_visualization_module.py",
            "execute/test_execute_pymdp_visualizer.py",
        ],
        "markers": [],
        "timeout_seconds": 300,
        "max_failures": 10,
        "parallel": True,
    },
    "llm": {
        "name": "LLM Module Tests",
        "description": "LLM integration and analysis tests",
        "files": [
            "llm/test_llm_ollama.py",
            "llm/test_llm_ollama_integration.py",
            "llm/test_llm_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "ontology": {
        "name": "Ontology Module Tests",
        "description": "Ontology processing and validation tests",
        "files": [
            "ontology/test_ontology_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "website": {
        "name": "Website Module Tests",
        "description": "Website generation tests",
        "files": [
            "website/test_website_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "report": {
        "name": "Report Module Tests",
        "description": "Report generation and formatting tests",
        "files": [
            "report/test_report_formats.py",
            "report/test_report_generation.py",
            "report/test_report_integration.py",
            "report/test_report_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "environment": {
        "name": "Environment Module Tests",
        "description": "Environment setup and validation tests",
        "files": [
            "test_environment_dependencies.py",
            "test_environment_integration.py",
            "test_environment_overall.py",
            "test_environment_python.py",
            "test_environment_system.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "type_checker": {
        "name": "Type Checker Module Tests",
        "description": "Type checking and validation tests",
        "files": [
            "type_checker/test_type_checker_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "validation": {
        "name": "Validation Module Tests",
        "description": "Validation and consistency tests",
        "files": [
            "validation/test_validation_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "model_registry": {
        "name": "Model Registry Module Tests",
        "description": "Model registry and versioning tests",
        "files": [
            "model_registry/test_model_registry_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "analysis": {
        "name": "Analysis Module Tests",
        "description": "Analysis and statistical tests",
        "files": [
            "analysis/test_analysis_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "integration": {
        "name": "Integration Module Tests",
        "description": "System integration tests",
        "files": [
            "integration/test_integration_overall.py",
            "integration/test_integration_processor.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "security": {
        "name": "Security Module Tests",
        "description": "Security validation tests",
        "files": [
            "security/test_security_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "research": {
        "name": "Research Module Tests",
        "description": "Research tools tests",
        "files": [
            "research/test_research_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "ml_integration": {
        "name": "ML Integration Module Tests",
        "description": "Machine learning integration tests",
        "files": [
            "ml_integration/test_ml_integration_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "advanced_visualization": {
        "name": "Advanced Visualization Module Tests",
        "description": "Advanced visualization tests",
        "files": [
            "advanced_visualization/test_advanced_visualization_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True,
    },
    "gui": {
        "name": "GUI Module Tests",
        "description": "GUI functionality and widget tests",
        "files": [
            "gui/test_gui_functionality.py",
            "gui/test_gui_overall.py",
        ],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True,
    },
    "comprehensive": {
        "name": "Comprehensive API Tests",
        "description": "Comprehensive API and integration tests",
        "files": [
            "api/test_comprehensive_api.py",
            "pipeline/test_main_orchestrator.py",
            "test_core_modules.py",
            "test_coverage_assessment.py",
            "test_coverage_overall.py",
            "test_fast_suite.py",
            "test_performance_overall.py",
            "test_runner_helper.py",
            "test_unit_overall.py",
            "utils/test_new_utils.py",
        ],
        "markers": [],
        "timeout_seconds": 300,
        "max_failures": 15,
        "parallel": False,
    },
}


def get_category_names() -> List[str]:
    """Get list of all category names."""
    return list(MODULAR_TEST_CATEGORIES.keys())


def get_category(name: str) -> TestCategory:
    """Get category configuration by name (empty dict when unknown)."""
    return MODULAR_TEST_CATEGORIES.get(name, {})


def get_category_files(name: str) -> List[str]:
    """Get test files for a specific category."""
    category = MODULAR_TEST_CATEGORIES.get(name, {})
    return list(category.get("files", []))


def get_all_test_files() -> List[str]:
    """Get all test files across all categories (sorted, deduplicated)."""
    files: set[str] = set()
    for category in MODULAR_TEST_CATEGORIES.values():
        files.update(category.get("files", []))
    return sorted(files)


def missing_category_files(
    test_dir: str | Path | None = None,
) -> Dict[str, List[str]]:
    """Report category files that do not exist under ``test_dir``.

    Args:
        test_dir: Directory the category file lists are relative to
            (defaults to the ``src/tests/`` directory containing this file).

    Returns:
        Mapping of category name to the category's file entries that are
        absent on disk. Categories with no missing files are omitted.
        Discovery skips missing entries, so this is a drift detector for
        the routing table, not an error by itself.
    """
    base = Path(test_dir) if test_dir is not None else Path(__file__).parent
    missing: dict[str, list[str]] = {}
    for name, category in MODULAR_TEST_CATEGORIES.items():
        absent = [
            entry
            for entry in category.get("files", [])
            if not (base / entry).exists() and not list(base.glob(entry))
        ]
        if absent:
            missing[name] = absent
    return missing


__all__: list[str] = [
    "TestCategory",
    "MODULAR_TEST_CATEGORIES",
    "get_category_names",
    "get_category",
    "get_category_files",
    "get_all_test_files",
    "missing_category_files",
]
