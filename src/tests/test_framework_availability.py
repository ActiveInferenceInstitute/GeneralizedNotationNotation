#!/usr/bin/env python3
"""Tests for src/utils/framework_availability.py.

Zero-mock: every check is verified against the real interpreter's
``importlib.util.find_spec`` results, not mocked-out stubs.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.framework_availability import (  # noqa: E402
    FRAMEWORK_IMPORT_CHECK,
    FrameworkStatus,
    check_framework,
    is_framework_available,
)


@pytest.mark.parametrize("framework", list(FRAMEWORK_IMPORT_CHECK.keys()))
def test_is_framework_available_matches_find_spec(framework):
    module_name, _ = FRAMEWORK_IMPORT_CHECK[framework]
    expected = importlib.util.find_spec(module_name) is not None
    assert is_framework_available(framework) is expected


def test_unknown_framework_returns_true():
    # Legacy fall-through semantics from execute/processor.py:115-116
    assert is_framework_available("definitely-not-a-framework") is True


def test_check_framework_returns_structured_status_when_available():
    # pick any framework that IS available in the local venv (pymdp is required core)
    status = check_framework("pymdp")
    assert isinstance(status, FrameworkStatus)
    assert status.name == "pymdp"
    if status.available:
        assert status.missing_module is None
        assert status.install_hint is None
    else:
        assert status.missing_module == "pymdp"
        assert status.install_hint is not None


def test_check_framework_populates_hint_when_unavailable():
    # Deliberately unknown module: use a unique entry that will never be installed.
    # Patch FRAMEWORK_IMPORT_CHECK locally via the public mutable dict.
    FRAMEWORK_IMPORT_CHECK["__nonexistent_test_fw__"] = (
        "__module_that_cannot_exist_abc123__",
        "do not install — test only",
    )
    try:
        status = check_framework("__nonexistent_test_fw__")
        assert status.available is False
        assert status.missing_module == "__module_that_cannot_exist_abc123__"
        assert status.install_hint == "do not install — test only"
    finally:
        FRAMEWORK_IMPORT_CHECK.pop("__nonexistent_test_fw__", None)


def test_unknown_framework_has_available_true_in_status():
    status = check_framework("definitely-not-a-framework")
    assert status.available is True
    assert status.missing_module is None


def test_framework_import_check_includes_required_runners():
    # Regression guard: Phase 0.2 must include every runner from executor.py.
    # Keep this list aligned with src/execute/executor.py framework runners.
    required = {"jax", "numpyro", "pytorch", "discopy", "bnlearn", "pymdp"}
    assert required.issubset(FRAMEWORK_IMPORT_CHECK.keys())
