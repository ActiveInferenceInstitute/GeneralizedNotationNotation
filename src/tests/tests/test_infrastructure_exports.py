"""Contracts for the ``tests.infrastructure`` re-export surface."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.fast


def test_infrastructure_exports_are_importable() -> None:
    """Every ``__all__`` entry resolves — guards the surface utils/test_utils.py uses."""
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

    stdout = "ERROR collecting src/tests/x.py\nE: ImportError: nope\n" * 2
    errors = extract_collection_errors(stdout, "")
    assert len(errors) == len(set(errors))
    assert errors


def test_check_test_dependencies_reports_pytest() -> None:
    import logging

    from tests.infrastructure import check_test_dependencies

    deps = check_test_dependencies(logging.getLogger("contract-test"))
    assert deps["pytest"] is True
    assert "psutil" in deps
