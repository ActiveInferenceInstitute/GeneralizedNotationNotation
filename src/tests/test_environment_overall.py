"""
Test Environment Overall Tests

This file contains comprehensive tests for the environment module functionality.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEnvironmentModuleComprehensive:
    """Comprehensive tests for the environment module."""

    @pytest.mark.unit
    def test_environment_module_imports(self) -> None:
        """Test that environment module can be imported."""
        import setup

        assert hasattr(setup, "__version__")
        assert hasattr(setup, "EnvironmentManager")
        assert hasattr(setup, "VirtualEnvironment")
        assert hasattr(setup, "get_environment_info")

    @pytest.mark.unit
    def test_environment_manager_instantiation(self) -> None:
        """Test EnvironmentManager class instantiation."""
        from setup import EnvironmentManager

        manager = EnvironmentManager()
        assert manager is not None
        assert hasattr(manager, "setup_environment")
        assert hasattr(manager, "validate_environment")

    @pytest.mark.unit
    def test_virtual_environment_instantiation(self) -> None:
        """Test VirtualEnvironment class instantiation."""
        from setup import VirtualEnvironment

        venv = VirtualEnvironment("test_env")
        assert venv is not None
        assert hasattr(venv, "create")
        assert hasattr(venv, "activate")

    @pytest.mark.unit
    def test_environment_module_info(self) -> None:
        """Test environment module information retrieval."""
        from setup import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info
        assert "environment_types" in info

    @pytest.mark.unit
    def test_environment_validation(self) -> None:
        """Test environment validation functionality."""
        from setup import validate_environment

        result = validate_environment()
        assert isinstance(result, dict)
        assert "python_version" in result
        assert "dependencies" in result


class TestEnvironmentFunctionality:
    """Tests for environment functionality."""

    @pytest.mark.unit
    def test_environment_setup(self, isolated_temp_dir: Any) -> None:
        """Test environment setup functionality."""
        from setup import EnvironmentManager

        manager = EnvironmentManager()
        result = manager.setup_environment(isolated_temp_dir)
        assert result is not None

    @pytest.mark.slow
    @pytest.mark.unit
    def test_dependency_installation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dependency installation must request the maintained dev test extras."""
        from setup import install_dependencies, uv_management

        commands: list[list[str]] = []

        def fake_run(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            commands.append([str(part) for part in command])
            return subprocess.CompletedProcess(
                command, 0, stdout="Python 3.12.0", stderr=""
            )

        monkeypatch.setattr(uv_management.subprocess, "run", fake_run)
        monkeypatch.setattr(
            uv_management,
            "get_installed_package_versions",
            lambda verbose=False: {},
        )

        result = install_dependencies(verbose=False, dev=True)
        assert isinstance(result, bool)
        assert result is True
        assert any(command[1:4] == ["sync", "--extra", "dev"] for command in commands)

    @pytest.mark.unit
    def test_python_version_check(self) -> None:
        """Test Python version checking."""
        from setup import check_python_version

        result = check_python_version()
        assert isinstance(result, bool)


class TestEnvironmentIntegration:
    """Integration tests for environment module."""

    @pytest.mark.integration
    def test_environment_pipeline_integration(self, isolated_temp_dir: Any) -> None:
        """Test environment module integration with pipeline."""
        from setup import EnvironmentManager

        manager = EnvironmentManager()
        result = manager.validate_environment()
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.integration
    def test_environment_mcp_integration(self) -> None:
        """Test environment MCP integration."""
        from setup.mcp import register_tools

        assert callable(register_tools)


def test_environment_module_completeness() -> None:
    """Test that environment module has all required components."""
    required_components: list[Any] = [
        "EnvironmentManager",
        "VirtualEnvironment",
        "get_module_info",
        "validate_environment",
        "install_dependencies",
        "check_python_version",
    ]
    try:
        import setup

        for component in required_components:
            assert hasattr(setup, component), f"Missing component: {component}"
    except ImportError as exc:
        raise AssertionError("Environment module must be importable") from exc


@pytest.mark.slow
def test_environment_module_performance() -> None:
    """Test environment module performance characteristics."""
    import time

    from setup import EnvironmentManager

    manager = EnvironmentManager()
    start_time = time.time()
    manager.validate_environment()
    processing_time = time.time() - start_time
    assert processing_time < 10.0
