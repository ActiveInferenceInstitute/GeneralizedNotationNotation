"""Real-behavior tests for ``src/gnn/pipeline/health_check.py``.

The enhanced health checker is documented pipeline API
(``gnn.pipeline.__all__`` exports ``EnhancedHealthChecker`` and
``run_enhanced_health_check``; usage examples live in the module's
SKILL.md), but had no direct tests. These pin the check semantics that
the health score is built from.

Environment-sensitivity note: the dependency/group assertions below hold
for the committed ``uv.lock`` dev environment (core install has
numpy/matplotlib/networkx/pandas/pytest/pymdp/openai/plotly/seaborn;
gradio/anthropic/bokeh are not in any default extra).
"""

from __future__ import annotations

from typing import Any

import pytest

import gnn.pipeline.health_check as health_check_module
from gnn.pipeline.health_check import EnhancedHealthChecker, run_enhanced_health_check


def test_core_dependencies_healthy_in_dev_environment() -> None:
    """Every core dependency is importable in the committed dev env.

    Regression guard: ``pyyaml`` was listed by its distribution name, but
    the module imports as ``yaml`` — core dependencies therefore reported
    ``unhealthy`` on every machine before the fix.
    """
    checker = EnhancedHealthChecker()

    results = checker.check_core_dependencies()

    assert checker.core_dependencies == {
        "numpy": ">=1.21.0",
        "matplotlib": ">=3.5.0",
        "networkx": ">=2.6.0",
        "pandas": ">=1.3.0",
        "pytest": ">=6.0.0",
        "yaml": ">=6.0",
    }
    assert results["status"] == "healthy"
    assert results["missing"] == []
    assert results["version_issues"] == []
    assert results["total_checked"] == len(checker.core_dependencies)


def test_core_dependencies_reports_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module that cannot import lands in `missing` and flips status."""
    real_import_module = health_check_module.importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yaml":  # the import name of the PyYAML distribution
            raise ImportError(f"No module named {name!r}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(health_check_module.importlib, "import_module", fake_import)

    results = EnhancedHealthChecker().check_core_dependencies()

    assert results["status"] == "unhealthy"
    assert any(dep.startswith("yaml") for dep in results["missing"])
    assert len(results["missing"]) == 1


def test_optional_dependency_group_statuses() -> None:
    """Group status derives from per-dependency availability."""
    results = EnhancedHealthChecker().check_optional_dependencies()

    # pymdp is a core dependency -> simulation group fully available.
    simulation = results["simulation"]
    assert simulation["status"] == "available"
    assert simulation["critical"] is True

    # openai is core, anthropic is not installed -> partial.
    llm = results["llm"]
    assert llm["status"] == "partial"
    assert "anthropic" in llm["missing"]
    assert any(dep.startswith("openai") for dep in llm["available"])

    # gradio is not installed -> gui group unavailable.
    assert results["gui"]["status"] == "unavailable"

    # bokeh is not installed, plotly/seaborn are -> partial.
    visualization = results["visualization"]
    assert visualization["status"] == "partial"
    assert "bokeh" in visualization["missing"]


def test_check_system_resources_limited_without_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without psutil the resource check degrades to a limited report."""
    monkeypatch.setattr(health_check_module, "PSUTIL_AVAILABLE", False)

    results = EnhancedHealthChecker().check_system_resources()

    assert results["status"] == "limited"
    assert "psutil" in results["error"]
    assert results["cpu"] == {}
    assert results["memory"] == {}


def test_check_pipeline_structure_complete_on_current_layout() -> None:
    """All 25 numbered orchestrators exist at src/gnn/N_*.py.

    The structure check globs ``src/gnn/{step}_*.py``; the package
    restructure moved the orchestrators there (not away), so the check
    must report a complete structure on the current tree.
    """
    results = EnhancedHealthChecker().check_pipeline_structure()

    assert results["status"] == "complete"
    assert results["missing_scripts"] == []
    # 25 numbered orchestrators + main.py + __init__.py
    assert len(results["available_scripts"]) == 27
    assert "main.py" in results["available_scripts"]


def test_run_enhanced_health_check_returns_full_report() -> None:
    """The top-level entry point assembles every check into one report."""
    results = run_enhanced_health_check()

    expected_keys = {
        "system_resources",
        "core_dependencies",
        "optional_dependencies",
        "pipeline_structure",
        "pipeline_integration",
        "health_score",
        "execution_time",
        "recommendations",
    }
    assert expected_keys <= set(results)
    assert isinstance(results["health_score"]["score"], (int, float))
    assert 0 <= results["health_score"]["score"] <= 100
