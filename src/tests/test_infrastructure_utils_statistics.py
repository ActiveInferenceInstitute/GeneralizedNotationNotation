"""Unit tests for pytest output and coverage parsing helpers."""

import pytest

from tests.infrastructure.report_generator import flatten_pipeline_test_summary
from tests.infrastructure.utils import parse_coverage_statistics, parse_test_statistics

pytestmark = pytest.mark.fast


def test_parse_test_statistics_quiet_summary() -> None:
    out = "\n===== 10 passed, 2 skipped in 1.2s =====\n"
    s = parse_test_statistics(out)
    assert s["passed"] == 10
    assert s["skipped"] == 2
    assert s["failed"] == 0
    assert s["total"] == 12
    assert s["tests_run"] == 12


def test_parse_test_statistics_with_failed_and_errors() -> None:
    out = "foo\n22 passed, 5 failed, 3 skipped, 1 error in 9.99s\n"
    s = parse_test_statistics(out)
    assert s["passed"] == 22
    assert s["failed"] == 5
    assert s["skipped"] == 3
    assert s["errors"] == 1
    assert s["total"] == 31


def test_parse_test_statistics_verbose_per_node() -> None:
    out = """
src/tests/foo.py::test_a PASSED
src/tests/foo.py::test_b FAILED
"""
    s = parse_test_statistics(out)
    assert s["passed"] == 1
    assert s["failed"] == 1
    assert s["total"] == 2


def test_parse_coverage_statistics_none() -> None:
    assert parse_coverage_statistics(None) == {}


def test_parse_coverage_statistics_rejects_multiline_string() -> None:
    assert parse_coverage_statistics("line1\nline2") == {}


def test_flatten_pipeline_test_summary() -> None:
    flat = flatten_pipeline_test_summary(
        {
            "execution_summary": {
                "tests_run": 100,
                "tests_passed": 95,
                "tests_failed": 3,
                "tests_skipped": 2,
            }
        }
    )
    assert flat["total_tests_run"] == 100
    assert flat["total_tests_passed"] == 95
    assert flat["total_tests_failed"] == 3
    assert flat["total_tests_skipped"] == 2
    assert flat["success_rate"] == 95.0
