"""Contracts for the ``tests.infrastructure`` re-export surface."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.fast


def test_infrastructure_exports_are_importable() -> None:
    """Every ``__all__`` entry resolves — guards the surface utils/testing_utils.py uses."""
    infrastructure = importlib.import_module("tests.infrastructure")
    for name in infrastructure.__all__:
        assert hasattr(infrastructure, name), f"tests.infrastructure.{name} missing"


def test_test_execution_config_defaults() -> None:
    from tests.infrastructure import TestExecutionConfig, TestExecutionResult

    config = TestExecutionConfig()
    assert config.timeout_seconds > 0
    assert config.max_failures > 0
    assert config.memory_limit_mb > 0

    result = TestExecutionResult(
        success=True,
        tests_run=3,
        tests_passed=2,
        tests_failed=1,
        tests_skipped=0,
        execution_time=1.5,
        memory_peak_mb=100.0,
    )
    assert result.to_dict()["tests_failed"] == 1


def test_flatten_pipeline_test_summary_zero_tests() -> None:
    from tests.infrastructure.report_generator import flatten_pipeline_test_summary

    flat = flatten_pipeline_test_summary({"execution_summary": {}})
    assert flat["total_tests_run"] == 0
    assert flat["success_rate"] == 0.0


def test_extract_collection_errors_dedupes() -> None:
    from tests.infrastructure import extract_collection_errors

    stdout = "ERROR collecting tests/x.py\nE: ImportError: nope\n" * 2
    errors = extract_collection_errors(stdout, "")
    assert len(errors) == len(set(errors))
    assert errors


def test_check_test_dependencies_reports_pytest() -> None:
    import logging

    from tests.infrastructure import check_test_dependencies

    deps = check_test_dependencies(logging.getLogger("contract-test"))
    assert deps["pytest"] is True
    assert "psutil" in deps


# I3 (S2-33 Step 0): the 113 ``gnn.utils._EXPORT_MAP`` keys are a frozen
# public surface. The concern-package split repoints map *values* to new leaf
# modules; keys may never be removed (that breaks external
# ``from gnn.utils import X`` consumers) or added (surface growth needs an
# owner decision). This literal is checked in precisely so a failing
# assertion cannot be made green without a deliberate, reviewable edit.
GOLDEN_EXPORT_MAP_KEYS: frozenset[str] = frozenset(
    {
        "ArgumentParser",
        "BaseProcessor",
        "COVERAGE_TARGETS",
        "CoverageTarget",
        "DependencySpec",
        "DependencyValidator",
        "ErrorCategory",
        "ErrorCodeRegistry",
        "ErrorContext",
        "ErrorRecoveryManager",
        "ErrorSeverity",
        "ExitCode",
        "GNNPipelineConfig",
        "LLMConfig",
        "ModelConfig",
        "OntologyConfig",
        "PerformanceTracker",
        "PipelineArguments",
        "PipelineConfig",
        "PipelineErrorHandler",
        "PipelineErrorSeverity",
        "PipelineLogger",
        "ProcessingResult",
        "RecoveryArgumentParser",
        "RecoveryStrategy",
        "SAPFConfig",
        "SetupConfig",
        "StepConfiguration",
        "StructuredLogger",
        "TEST_CATEGORIES",
        "TEST_CONFIG",
        "TEST_STAGES",
        "TestCategory",
        "TestResult",
        "TestRunner",
        "TestStage",
        "TypeCheckerConfig",
        "WebsiteConfig",
        "build_step_command_args",
        "check_optional_dependencies",
        "cleanup_test_environment",
        "create_processor",
        "create_standardized_pipeline_script",
        "execute_pipeline_step_template",
        "format_and_log_error",
        "format_error_message",
        "generate_correlation_id",
        "generate_pipeline_health_report",
        "generate_test_report",
        "get_config_value",
        "get_current_memory_usage",
        "get_dependency_status",
        "get_output_dir_for_script",
        "get_performance_summary",
        "get_pipeline_logger",
        "get_pipeline_step_info",
        "get_pipeline_utilities",
        "get_recovery_manager",
        "get_system_info",
        "get_test_artifacts",
        "get_test_configuration",
        "get_test_coverage",
        "get_test_dependencies",
        "get_test_duration",
        "get_test_environment",
        "get_test_logs",
        "get_test_metadata",
        "get_test_performance",
        "get_test_progress",
        "get_test_results",
        "get_test_statistics",
        "get_test_status",
        "get_test_summary",
        "get_test_timestamps",
        "get_venv_python",
        "handle_file_system_error",
        "handle_network_error",
        "handle_timeout_error",
        "install_missing_dependencies",
        "install_test_dependencies",
        "load_config",
        "log_pipeline_complete",
        "log_pipeline_start",
        "log_section_header",
        "log_step_error",
        "log_step_start",
        "log_step_success",
        "log_step_warning",
        "parse_arguments",
        "performance_tracker",
        "run_test_category",
        "run_test_stage",
        "run_tests",
        "save_config",
        "set_config_value",
        "set_correlation_context",
        "setup_correlation_context",
        "setup_main_logging",
        "setup_step_logging",
        "setup_test_environment",
        "track_operation_standalone",
        "validate_and_convert_paths",
        "validate_config",
        "validate_coverage_targets",
        "validate_output_directory",
        "validate_pipeline_configuration",
        "validate_pipeline_dependencies",
        "validate_pipeline_dependencies_if_available",
        "validate_pipeline_step_sequence",
        "validate_step_prerequisites",
        "validate_test_configuration",
        "validate_test_dependencies",
        "validate_test_environment",
    }
)


def test_export_map_surface_is_frozen() -> None:
    """``_EXPORT_MAP`` stays exactly the checked-in 113-key golden surface.

    S2-33 I3: prose counts drifted before (the recurring 111-vs-113 drift);
    the key set itself is now asserted instead of restated in prose. Removals
    break external ``from gnn.utils import X`` consumers; additions are
    surface growth that needs an owner decision.
    """
    from gnn.utils import _EXPORT_MAP
    from gnn.utils import __all__ as export_names

    keys = set(_EXPORT_MAP)
    assert len(_EXPORT_MAP) == 113
    missing = GOLDEN_EXPORT_MAP_KEYS - keys
    assert not missing, (
        f"_EXPORT_MAP keys removed from the frozen surface: {sorted(missing)}"
    )
    added = keys - GOLDEN_EXPORT_MAP_KEYS
    assert not added, (
        f"_EXPORT_MAP keys added (surface growth needs owner sign-off): {sorted(added)}"
    )
    # The unique ``__all__`` names stay pinned to the same golden surface:
    # 118 entries = the 113 map keys + UTILS_AVAILABLE, with the four
    # log_step_* names deliberately double-listed across the logging and
    # structured-logging sections.
    assert set(export_names) - {"UTILS_AVAILABLE"} == keys
