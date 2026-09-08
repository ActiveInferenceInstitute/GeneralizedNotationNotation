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


def test_core_dependencies_reports_version_unknown() -> None:
    """A dep whose import returns __version__=='unknown' for a non-trivial
    version requirement lands in ``version_issues`` (not ``missing``)."""
    real_import_module = health_check_module.importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pytest":

            class _FakeMod:
                __version__ = "unknown"

            return _FakeMod()
        return real_import_module(name, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(health_check_module.importlib, "import_module", fake_import)
    try:
        results = EnhancedHealthChecker().check_core_dependencies()
    finally:
        monkeypatch.undo()

    assert results["status"] == "healthy"
    assert any(
        "pytest" in issue and "unknown" in issue for issue in results["version_issues"]
    )
    assert "pytest >=6.0.0" not in results["missing"]


def test_optional_dependencies_julia_subprocess_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia group status derives from the ``julia --version`` subprocess:
    returncode 0 with stdout -> available; returncode != 0 -> missing;
    FileNotFoundError -> missing (group stays missing -> unavailable)."""

    class _Result:
        def __init__(self, returncode: int, stdout: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        return _Result(0, "1.10.0\n")

    monkeypatch.setattr(health_check_module.subprocess, "run", fake_run)
    results = EnhancedHealthChecker().check_optional_dependencies()
    julia = results["julia"]
    assert any("1.10.0" in entry for entry in julia["available"])

    def fake_run_fail(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        return _Result(1)

    monkeypatch.setattr(health_check_module.subprocess, "run", fake_run_fail)
    results = EnhancedHealthChecker().check_optional_dependencies()
    assert "julia" in results["julia"]["missing"]

    def fake_run_missing(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "no julia")

    monkeypatch.setattr(health_check_module.subprocess, "run", fake_run_missing)
    results = EnhancedHealthChecker().check_optional_dependencies()
    assert "julia" in results["julia"]["missing"]


def test_pipeline_structure_reports_incomplete_when_script_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing numbered orchestrator flips status to ``incomplete`` and
    records the missing stem — the recovery the happy-path test does not
    exercise."""
    checker = EnhancedHealthChecker()

    # Point the module's __file__ at a tmp tree with no numbered scripts so
    # ``Path(__file__).parent.parent`` resolves to an empty parent and the
    # ``0_*.py`` ... ``24_*.py`` globs all miss (plus main.py/__init__.py).
    monkeypatch.setattr(
        health_check_module, "__file__", str(tmp_path / "health_check.py")
    )
    results = checker.check_pipeline_structure()

    assert results["status"] == "incomplete"
    assert len(results["missing_scripts"]) >= 1


def test_pipeline_integration_limited_when_flag_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With PIPELINE_INTEGRATION False the integration check degrades to
    ``limited`` without touching the real pipeline modules."""
    monkeypatch.setattr(health_check_module, "PIPELINE_INTEGRATION", False)
    results = EnhancedHealthChecker().check_pipeline_integration()
    assert results["integration_status"] == "limited"


def test_pipeline_integration_partial_on_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A get_pipeline_config() probe failure flips integration to ``partial``.

    health_check binds ``get_pipeline_config`` at import time, so the patch
    target is the module-local name, not the canonical ``gnn.pipeline.config``.
    """

    def boom(*args: Any, **kwargs: Any) -> "dict[str, Any]":
        raise RuntimeError("probe failure")

    monkeypatch.setattr(health_check_module, "get_pipeline_config", boom)
    results = EnhancedHealthChecker().check_pipeline_integration()
    assert results["integration_status"] == "partial"
    assert results["config_available"] is False


def test_calculate_overall_health_scoring_tiers() -> None:
    """Scoring: healthy core/structure/system + full integration + at least
    one optional group yields an ``excellent`` rating; empty results yield
    ``poor``; partial core + missing scripts + partial integration yields a
    mid-band score with the partial-credit branches exercised."""
    checker = EnhancedHealthChecker()

    # Empty results -> poor.
    checker.results = {}
    assert checker._calculate_overall_health()["rating"] == "poor"

    # Healthy everything -> excellent.
    checker.results = {
        "system_resources": {"status": "healthy"},
        "core_dependencies": {"status": "healthy", "missing": []},
        "pipeline_structure": {"status": "complete", "missing_scripts": []},
        "pipeline_integration": {"integration_status": "full"},
        "optional_dependencies": {
            "simulation": {"status": "available"},
            "llm": {"status": "available"},
            "gui": {"status": "unavailable"},
        },
    }
    score = checker._calculate_overall_health()
    assert score["rating"] == "excellent"
    # Optional cap: 2 available groups * 2 = 4/10 -> 20+30+25+15+4 = 94.
    assert score["score"] == 94.0

    # Partial credit: 2 missing core deps (<=2), a few missing scripts (<=3),
    # partial integration, no optional groups.
    checker.results = {
        "system_resources": {"status": "healthy"},
        "core_dependencies": {"status": "unhealthy", "missing": ["a", "b"]},
        "pipeline_structure": {
            "status": "incomplete",
            "missing_scripts": ["1", "2", "3"],
        },
        "pipeline_integration": {"integration_status": "partial"},
        "optional_dependencies": {},
    }
    score = checker._calculate_overall_health()
    # 20 + 30*0.7 + 25*0.8 + 15*0.5 + 0 = 20+21+20+7.5 = 68.5/100 -> fair
    assert score["rating"] == "fair"
    assert 60.0 <= score["score"] < 75.0


def test_main_exit_code_maps_rating_to_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: "Any"
) -> None:
    """main() returns 0 for good/excellent, 1 for fair, 2 for poor; --json
    emits a JSON document to stdout."""
    import json as _json
    import sys

    poor_report = {
        "health_score": {"rating": "poor", "score": 10.0},
        "system_resources": {},
        "core_dependencies": {"available": [], "missing": []},
        "pipeline_structure": {"available_scripts": []},
    }
    fair_report = {
        "health_score": {"rating": "fair", "score": 65.0},
        "system_resources": {},
        "core_dependencies": {"available": [], "missing": []},
        "pipeline_structure": {"available_scripts": []},
    }

    monkeypatch.setattr(
        health_check_module, "run_enhanced_health_check", lambda _v: poor_report
    )
    monkeypatch.setattr(sys, "argv", ["health_check", "--json"])
    import io

    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    assert health_check_module.main() == 2
    payload = _json.loads(buf.getvalue())
    assert payload["health_score"]["rating"] == "poor"

    monkeypatch.setattr(
        health_check_module, "run_enhanced_health_check", lambda _v: fair_report
    )
    monkeypatch.setattr(sys, "argv", ["health_check", "--json"])
    buf2 = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf2)
    assert health_check_module.main() == 1
    assert _json.loads(buf2.getvalue())["health_score"]["rating"] == "fair"
