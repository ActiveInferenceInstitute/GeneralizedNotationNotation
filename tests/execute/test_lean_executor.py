"""Regression tests for the ``lean`` executor framework registration."""

from __future__ import annotations

import pytest

import gnn.execute.executor as executor_module
from gnn.execute.executor import (
    ExecutorFrameworkSpec,
    GNNExecutor,
    _framework_specs,
    list_frameworks,
)


def test_executor_registers_lean() -> None:
    frameworks = {entry["framework"]: entry for entry in list_frameworks()}
    assert "lean" in frameworks
    assert frameworks["lean"]["result_key"] == "lean_executions"
    assert frameworks["lean"]["operation"] == "execute_lean_verification"


def test_lean_spec_present() -> None:
    specs = [spec for spec in _framework_specs() if spec.framework_dir_key == "lean"]
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ExecutorFrameworkSpec)
    assert spec.result_key == "lean_executions"
    assert "FEP_LEAN_ROOT" in spec.unavailable_log
    assert "FEP_LEAN_ROOT" in spec.unavailable_message


def test_executor_lean_unavailable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no resolvable fep_lean checkout, the executor reports unavailable."""
    from gnn.execute.lean.lean_runner import FEP_LEAN_ROOT_ENV

    monkeypatch.setenv(FEP_LEAN_ROOT_ENV, "/nonexistent/fep_lean")
    from gnn.execute.lean.lean_runner import resolve_fep_lean_root

    assert resolve_fep_lean_root() is None

    monkeypatch.setattr(executor_module, "LEAN_AVAILABLE", False)
    result = GNNExecutor()._execute_lean_verification("model.md")
    assert result == {"success": False, "error": "fep_lean unavailable"}
