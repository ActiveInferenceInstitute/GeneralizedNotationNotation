"""Contracts for the unified ``TestRunner`` (``tests.infrastructure.test_runner``)."""

from __future__ import annotations

import pytest

from tests.infrastructure import TestExecutionConfig
from tests.infrastructure.test_runner import TestRunner

pytestmark = pytest.mark.fast


def test_runner_module_has_single_source() -> None:
    """``tests.runner`` re-exports the same class as ``tests.infrastructure``."""
    from tests import runner as runner_mod
    from tests.infrastructure import TestRunner as canonical

    assert runner_mod.TestRunner is TestRunner
    assert runner_mod.TestRunner is canonical


def test_build_pytest_command_flags(tmp_path: object) -> None:
    import pathlib
    import sys
    from typing import cast

    config = TestExecutionConfig(
        max_failures=7, coverage=False, markers=["fast"], timeout_seconds=99
    )
    runner = TestRunner(config)
    cmd = runner._build_pytest_command(
        [pathlib.Path("src/tests/test_fast_suite.py")], cast("pathlib.Path", tmp_path)
    )
    assert cmd[:3] == [sys.executable, "-m", "pytest"]
    assert "--log-cli-level=WARNING" in cmd
    assert "--maxfail=7" in cmd
    assert "-m" in cmd and "fast" in cmd
    assert not any(arg.startswith("--cov") for arg in cmd)


def test_parse_pytest_output_summary_line() -> None:
    config = TestExecutionConfig()
    runner = TestRunner(config)
    stats = runner._parse_pytest_output(
        "===== 12 passed, 2 skipped in 1.20s =====\n", ""
    )
    assert stats["tests_passed"] == 12
    assert stats["tests_skipped"] == 2
    assert stats["tests_run"] == 14
    assert stats["success"] is True
    assert stats["collection_errors"] == []


def test_parse_pytest_output_failure_and_collection_errors() -> None:
    config = TestExecutionConfig()
    runner = TestRunner(config)
    stdout = "ERROR collecting src/tests/broken.py\nImportError: nope\n===== 1 failed in 0.1s =====\n"
    stats = runner._parse_pytest_output(stdout, "")
    assert stats["success"] is False
    assert stats["tests_failed"] == 1
    assert any("broken.py" in line for line in stats["collection_errors"])


def test_parse_pytest_output_zero_collected_is_failure() -> None:
    config = TestExecutionConfig()
    runner = TestRunner(config)
    stats = runner._parse_pytest_output("no tests ran in 0.01s\n", "")
    assert stats["success"] is False
    assert stats["tests_run"] == 0
