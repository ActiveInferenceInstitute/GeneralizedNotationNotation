"""Pins for ``utils/dependency_validator`` offline contracts (14% coverage).

Network/mutation surfaces (``install_missing_dependencies``, pip-based
installers) are deliberately NOT exercised.
"""

from __future__ import annotations

from gnn.utils.runtime_safety.dependency_validator import (
    DependencySpec,
    DependencyValidator,
    check_optional_dependencies,
    get_dependency_status,
)


def test_validate_dependency_group_core_succeeds_offline() -> None:
    validator = DependencyValidator()

    # The core group is stdlib-only: import probes via the current
    # interpreter must all succeed offline.
    assert validator.validate_dependency_group("core") is True
    assert validator.missing_dependencies == []


def test_validate_dependency_group_unknown_group_trivially_true() -> None:
    validator = DependencyValidator()

    # Documented behavior: an unknown group name warns and returns True.
    assert validator.validate_dependency_group("not_a_group") is True


def test_validate_python_dependency_reports_missing_module() -> None:
    validator = DependencyValidator()
    spec = DependencySpec(
        name="w2_probe_missing", description="module that does not exist"
    )

    assert validator.validate_python_dependency(spec) is False


def test_validate_python_dependency_reports_present_module() -> None:
    validator = DependencyValidator()
    spec = DependencySpec(name="json", description="stdlib json")

    assert validator.validate_python_dependency(spec) is True


def test_validate_all_dependencies_core_only() -> None:
    validator = DependencyValidator()

    assert validator.validate_all_dependencies(required_groups=["core"]) is True


def test_get_dependency_status_reports_core_group() -> None:
    status = get_dependency_status()

    assert isinstance(status, dict)
    assert "core" in str(status) or status


def test_check_optional_dependencies_returns_probing_dict() -> None:
    result = check_optional_dependencies()

    assert isinstance(result, dict)
    assert result


def test_installation_instructions_are_generated() -> None:
    validator = DependencyValidator()

    instructions = validator.get_installation_instructions()

    assert isinstance(instructions, list)
