"""Pins for ``utils/pipeline_dependencies`` (previously at 20% coverage)."""

from __future__ import annotations

from gnn.utils.pipeline_dependencies import (
    get_pipeline_dependency_manager,
)


def test_singleton_returns_same_manager() -> None:
    assert get_pipeline_dependency_manager() is get_pipeline_dependency_manager()


def test_check_dependency_available_module_reports_version_hint() -> None:
    manager = get_pipeline_dependency_manager()

    result = manager.check_dependency("json", use_cache=False)

    assert result.available is True
    assert result.install_hint == "uv pip install json"


def test_check_dependency_caches_results() -> None:
    manager = get_pipeline_dependency_manager()

    first = manager.check_dependency("pathlib", use_cache=True)
    assert "pathlib" in manager.dependency_cache
    assert manager.check_dependency("pathlib", use_cache=True) is first


def test_check_dependency_missing_module_reports_error() -> None:
    manager = get_pipeline_dependency_manager()

    result = manager.check_dependency("no_such_module_w2_probe", use_cache=False)

    assert result.available is False
    assert result.error
    assert result.install_hint == "uv pip install no_such_module_w2_probe"


def test_check_step_dependencies_core_step_succeeds() -> None:
    manager = get_pipeline_dependency_manager()

    result = manager.check_step_dependencies("3_gnn")

    # pathlib/json/logging are stdlib: the required set is fully met and
    # only the optional toml may degrade the step.
    assert result["status"] in {"healthy", "degraded"}
    assert result["errors"] == []
    assert all(dep.available for dep in result["required"].values())


def test_check_step_dependencies_unknown_step_fails_explicitly() -> None:
    manager = get_pipeline_dependency_manager()

    result = manager.check_step_dependencies("not_a_step")

    assert result["status"] == "unknown"


def test_graceful_import_yields_module_or_none() -> None:
    manager = get_pipeline_dependency_manager()

    with manager.graceful_import("json", "3_gnn") as module:
        assert module is not None
    with manager.graceful_import("no_such_module_w2_probe") as module:
        assert module is None


def test_dependency_report_covers_registered_steps() -> None:
    manager = get_pipeline_dependency_manager()

    report = manager.generate_dependency_report()

    assert report["summary"]["total_steps"] == len(manager.step_configs)
    assert set(report["steps"]) == set(manager.step_configs)
    assert report["summary"]["total_dependencies"] > 0
    assert (
        report["summary"]["available_dependencies"]
        + report["summary"]["missing_dependencies"]
        == report["summary"]["total_dependencies"]
    )
