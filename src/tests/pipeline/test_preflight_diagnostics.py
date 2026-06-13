"""Tests for preflight dependency diagnostics."""

from types import SimpleNamespace
from typing import Any

import pytest

from pipeline import preflight


def test_import_dependency_reports_direct_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(module_name: str) -> Any:
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(preflight, "import_module", fake_import_module)

    module, error = preflight._import_dependency("numpyro")

    assert module is None
    assert error == "missing"


def test_import_dependency_reports_nested_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(module_name: str) -> Any:
        raise ModuleNotFoundError("No module named 'jax.extend'", name="jax.extend")

    monkeypatch.setattr(preflight, "import_module", fake_import_module)

    module, error = preflight._import_dependency("numpyro")

    assert module is None
    assert error is not None
    assert error.startswith("import failed:")
    assert "jax.extend" in error


def test_check_environment_reports_step12_broken_import_as_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_dependency(module_name: str) -> tuple[Any | None, str | None]:
        if module_name == "numpyro":
            return None, "import failed: cannot import name 'xla_pmap_p'"
        return SimpleNamespace(__version__="test"), None

    monkeypatch.setattr(preflight, "_import_dependency", fake_import_dependency)
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(preflight.Path, "exists", lambda self: True)

    report = preflight.check_environment()

    assert report.checks_failed == 1
    assert any(
        issue.severity == "error"
        and "Package import failed: numpyro (step 12 backend)" in issue.message
        for issue in report.issues
    )
    assert not any("Package not found: numpyro" in issue.message for issue in report.issues)
