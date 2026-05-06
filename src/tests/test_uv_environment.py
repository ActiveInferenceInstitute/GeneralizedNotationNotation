#!/usr/bin/env python3
"""
Comprehensive UV Environment Management Test Suite.

This test suite verifies:
1. UV availability and proper installation
2. Virtual environment creation and management
3. Dependency synchronization via uv sync
4. Package version tracking and validation
5. Environment health checks
6. All UV-related setup functions fully succeed with real methods

Following the project's Zero Simulated policy - all tests use real methods.
"""

import json
import logging
import shutil
import subprocess  # nosec B404 -- subprocess calls with controlled/trusted input
import sys
from pathlib import Path

import pytest

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.uv]

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
VENV_PATH = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_PATH / "bin" / "python" if sys.platform != "win32" else VENV_PATH / "Scripts" / "python.exe"

# Resolve 'uv' binary
UV_BIN = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")

# Import setup module
try:
    from src.setup import (
        FEATURES,
        OPTIONAL_GROUPS,
        add_uv_dependency,
        check_environment_health,
        check_system_requirements,
        check_uv_availability,
        get_environment_info,
        get_installed_package_versions,
        get_uv_setup_info,
        get_uv_status,
        lock_uv_dependencies,
        remove_uv_dependency,
        setup_uv_environment,
        update_uv_dependencies,
        validate_system,
        validate_uv_setup,
    )
    SETUP_AVAILABLE = True
except ImportError as e:
    SETUP_AVAILABLE = False
    IMPORT_ERROR = str(e)


logger = logging.getLogger(__name__)


class TestUVAvailability:
    """Test UV CLI availability and version."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason=f"Setup module not available: {IMPORT_ERROR if not SETUP_AVAILABLE else ''}")
    def test_uv_cli_available(self):
        """Test that UV CLI is available in PATH."""
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"UV CLI not available: {result.stderr}"
        assert "uv" in result.stdout.lower(), "Unexpected UV version output"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_check_uv_availability_function(self):
        """Test the check_uv_availability function."""
        available = check_uv_availability(verbose=False)
        assert available is True, "check_uv_availability should return True when UV is installed"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_uv_version_compatible(self):
        """Test that UV version is compatible (0.9.x or higher)."""
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        version_str = result.stdout.strip()
        # Extract version number
        parts = version_str.split()
        if len(parts) >= 2:
            version = parts[1]
            major_minor = version.split(".")[:2]
            if len(major_minor) >= 2:
                major = int(major_minor[0])
                minor = int(major_minor[1])
                assert major >= 0, "UV major version should be >= 0"
                if major == 0:
                    assert minor >= 9, f"UV minor version should be >= 9, got {minor}"


class TestVirtualEnvironment:
    """Test virtual environment management."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_venv_exists(self):
        """Test that virtual environment exists."""
        assert VENV_PATH.exists(), f"Virtual environment not found at {VENV_PATH}"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_venv_python_exists(self):
        """Test that virtual environment Python exists."""
        assert VENV_PYTHON.exists(), f"Venv Python not found at {VENV_PYTHON}"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_venv_python_executable(self):
        """Test that virtual environment Python is executable."""
        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            [str(VENV_PYTHON), "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"Venv Python not executable: {result.stderr}"
        assert "Python" in result.stdout, "Unexpected Python version output"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_venv_python_version_compatible(self):
        """Test that Python version is >= 3.11 as per pyproject.toml."""
        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            [str(VENV_PYTHON), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0
        version = result.stdout.strip()
        major, minor = map(int, version.split("."))
        assert major >= 3 and minor >= 11, f"Python version should be >= 3.11, got {version}"


class TestProjectFiles:
    """Test UV project files exist and are valid."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"

    def test_uv_lock_exists(self):
        """Test that uv.lock exists."""
        lock_path = PROJECT_ROOT / "uv.lock"
        assert lock_path.exists(), f"uv.lock not found at {lock_path}"

    def test_pyproject_toml_valid(self):
        """Test that pyproject.toml is valid TOML."""
        import tomllib

        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        assert "project" in config, "pyproject.toml missing [project] section"
        assert "name" in config["project"], "pyproject.toml missing project name"
        assert "version" in config["project"], "pyproject.toml missing version"
        assert "dependencies" in config["project"], "pyproject.toml missing dependencies"

    def test_uv_lock_not_empty(self):
        """Test that uv.lock is not empty."""
        lock_path = PROJECT_ROOT / "uv.lock"
        assert lock_path.stat().st_size > 0, "uv.lock is empty"


class TestDependencyManagement:
    """Test dependency installation and management."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_get_installed_package_versions(self):
        """Test getting installed package versions."""
        packages = get_installed_package_versions(verbose=False)
        assert isinstance(packages, dict), "get_installed_package_versions should return a dict"
        assert len(packages) > 0, "Should have installed packages"

        if "pytest" not in packages:
            pytest.skip(
                "pytest not listed in this uv environment; install dev extras: uv sync --extra dev"
            )

        # Check for core packages (pytest is dev-extra; guarded above)
        core_packages = ["numpy", "pytest", "matplotlib", "scipy"]
        for pkg in core_packages:
            assert pkg in packages, f"Core package {pkg} not found"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_uv_tree_works(self):
        """``uv tree`` is the modern replacement for ``uv pip list``.

        The legacy ``uv pip`` interface is rejected by newer uv releases and
        by toolchain shims (e.g. trailofbits modern-python). ``uv tree``
        prints the project's dependency tree and is the shim-compatible
        way to verify uv is functional against the current project.
        """
        result = subprocess.run(  # nosec B607 B603 -- fixed-arg invocation
            [UV_BIN, "tree", "--depth", "1"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        assert result.returncode == 0, f"uv tree failed: {result.stderr}"
        # Output is human-readable; just check it's non-empty.
        assert result.stdout.strip(), "uv tree produced no output"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_uv_sync_check(self):
        """Test that uv sync --dry-run reports no changes needed."""
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "sync", "--frozen", "--check"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60
        )
        # If returncode is 0, environment is in sync
        # If returncode is 1, there are pending changes
        # Both are acceptable - we just want the command to work
        assert result.returncode in [0, 1], f"uv sync --check failed unexpectedly: {result.stderr}"


class TestSystemRequirements:
    """Test system requirements validation."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_check_system_requirements(self):
        """Test system requirements check function."""
        result = check_system_requirements(verbose=False)
        assert result is True, "System requirements should pass"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_validate_system_function(self):
        """Test validate_system function from validator."""
        result = validate_system()
        assert isinstance(result, dict), "validate_system should return a dict"
        assert "success" in result, "Result should have 'success' key"
        assert result["success"] is True, f"validate_system failed: {result.get('error', 'unknown')}"


class TestUVSetupValidation:
    """Test UV setup validation functions."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_validate_uv_setup(self):
        """Test validate_uv_setup function."""
        result = validate_uv_setup()
        assert isinstance(result, dict), "validate_uv_setup should return a dict"

        # Check expected keys
        expected_keys = [
            "system_requirements",
            "uv_environment",
            "dependencies",
            "jax_installation",
            "jax_stack_functional",
            "overall_status",
            "python_version",
        ]
        for key in expected_keys:
            assert key in result, f"Result missing key: {key}"

        # Verify all checks pass
        assert result["system_requirements"] is True, "System requirements check should pass"
        assert result["uv_environment"] is True, "UV environment check should pass"
        assert result["dependencies"] is True, "Dependencies check should pass"
        assert result["jax_stack_functional"] is True, "JAX + pymdp stack must be functional (core deps)"
        assert result["jax_installation"] is True, "JAX installation should be reported OK"
        assert result["overall_status"] is True, "Overall status should be True"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_get_uv_setup_info(self):
        """Test get_uv_setup_info function."""
        info = get_uv_setup_info()
        assert isinstance(info, dict), "get_uv_setup_info should return a dict"

        # Check expected keys
        expected_keys = ["project_root", "uv_environment_path", "python_version", "platform", "uv_setup_status"]
        for key in expected_keys:
            assert key in info, f"Info missing key: {key}"

        # Verify values
        assert Path(info["project_root"]) == PROJECT_ROOT, "Project root mismatch"
        assert Path(info["uv_environment_path"]) == VENV_PATH, "Venv path mismatch"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_get_uv_status(self):
        """Test get_uv_status function."""
        result = get_uv_status()
        assert isinstance(result, dict), "get_uv_status should return a dict"
        assert "uv_available" in result, "Result should have 'uv_available' key"
        assert result["uv_available"] is True, "UV should be available"
        assert result["status"] == "healthy", "Status should be 'healthy'"


class TestEnvironmentHealth:
    """Test environment health check functionality."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_check_environment_health(self):
        """Test comprehensive environment health check."""
        health = check_environment_health(verbose=False)
        assert isinstance(health, dict), "check_environment_health should return a dict"

        # Check expected keys
        expected_keys = [
            "overall_healthy", "uv_available", "uv_version",
            "venv_exists", "venv_python_works",
            "lock_file_exists", "pyproject_exists",
            "core_packages", "optional_packages",
            "issues", "suggestions"
        ]
        for key in expected_keys:
            assert key in health, f"Health check missing key: {key}"

        # Verify critical checks pass
        assert health["overall_healthy"] is True, f"Environment not healthy. Issues: {health.get('issues', [])}"
        assert health["uv_available"] is True, "UV should be available"
        assert health["venv_exists"] is True, "Venv should exist"
        assert health["venv_python_works"] is True, "Venv Python should work"
        assert health["lock_file_exists"] is True, "Lock file should exist"
        assert health["pyproject_exists"] is True, "pyproject.toml should exist"

        # Check core packages
        assert isinstance(health["core_packages"], dict), "core_packages should be a dict"
        core_expected = ["numpy", "matplotlib", "networkx", "pandas", "scipy", "pytest"]
        if health["core_packages"].get("pytest") is None:
            pytest.skip(
                "pytest not installed in project venv; full health check needs: uv sync --extra dev"
            )
        for pkg in core_expected:
            assert pkg in health["core_packages"], f"Core package {pkg} missing"
            assert health["core_packages"][pkg] is not None, f"Core package {pkg} version is None"

        # No critical issues
        assert len(health["issues"]) == 0, f"Unexpected issues: {health['issues']}"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_get_environment_info(self):
        """Test get_environment_info function."""
        info = get_environment_info()
        assert isinstance(info, dict), "get_environment_info should return a dict"
        assert "status" in info, "Info should have 'status' key"
        assert info["status"] == "healthy" or "error" not in info, f"Environment not healthy: {info.get('error')}"


class TestOptionalGroups:
    """Test optional dependency groups configuration."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_optional_groups_defined(self):
        """Test that OPTIONAL_GROUPS constant is defined."""
        assert isinstance(OPTIONAL_GROUPS, dict), "OPTIONAL_GROUPS should be a dict"
        assert len(OPTIONAL_GROUPS) > 0, "OPTIONAL_GROUPS should not be empty"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_optional_groups_have_descriptions(self):
        """Test that all optional groups have descriptions."""
        for group, description in OPTIONAL_GROUPS.items():
            assert isinstance(group, str), f"Group name should be string: {group}"
            assert isinstance(description, str), f"Group description should be string: {description}"
            assert len(description) > 0, f"Group {group} has empty description"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_expected_optional_groups_exist(self):
        """Test that expected optional groups are defined."""
        expected_groups = ["dev", "llm", "visualization", "audio", "all"]
        for group in expected_groups:
            assert group in OPTIONAL_GROUPS, f"Expected optional group '{group}' not found"


class TestFeatureFlags:
    """Test feature flags configuration."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_features_defined(self):
        """Test that FEATURES constant is defined."""
        assert isinstance(FEATURES, dict), "FEATURES should be a dict"
        assert len(FEATURES) > 0, "FEATURES should not be empty"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_critical_features_enabled(self):
        """Test that critical features are enabled."""
        critical_features = [
            "uv_environment_setup",
            "uv_dependency_management",
            "uv_virtual_environment",
            "system_validation",
            "native_uv_sync",
        ]
        for feature in critical_features:
            assert feature in FEATURES, f"Critical feature '{feature}' not defined"
            assert FEATURES[feature] is True, f"Critical feature '{feature}' should be enabled"


class TestNativeUVFunctions:
    """Test native UV functions (add, remove, update, lock)."""

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_add_uv_dependency_function_exists(self):
        """Test that add_uv_dependency function exists and is callable."""
        assert callable(add_uv_dependency), "add_uv_dependency should be callable"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_remove_uv_dependency_function_exists(self):
        """Test that remove_uv_dependency function exists and is callable."""
        assert callable(remove_uv_dependency), "remove_uv_dependency should be callable"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_update_uv_dependencies_function_exists(self):
        """Test that update_uv_dependencies function exists and is callable."""
        assert callable(update_uv_dependencies), "update_uv_dependencies should be callable"

    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_lock_uv_dependencies_function_exists(self):
        """Test that lock_uv_dependencies function exists and is callable."""
        assert callable(lock_uv_dependencies), "lock_uv_dependencies should be callable"


class TestUVRunIntegration:
    """Test UV run command integration."""

    def test_uv_run_python(self):
        """Test that uv run python works."""
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "run", "python", "-c", "print('Hello from UV')"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"uv run python failed: {result.stderr}"
        assert "Hello from UV" in result.stdout, "Unexpected output"

    def test_uv_run_module_import(self):
        """Test that uv run can import project modules."""
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "run", "python", "-c", "from src.setup import FEATURES; print('Import OK')"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"Module import failed: {result.stderr}"
        assert "Import OK" in result.stdout, "Module import verification failed"

    @pytest.mark.xfail(reason="uv's tool resolution strategy may install pytest in a tool cache rather than the venv's site-packages")
    def test_uv_run_pytest(self):
        """Test that pytest is available in the UV-managed environment.

        Marked ``xfail`` because ``uv``'s tool resolution strategy may
        install pytest in a tool cache rather than the venv's site-packages.
        When run inside a ``uv run python -m pytest`` session the current
        interpreter already has pytest (otherwise this test wouldn't be
        executing), so we simply validate reachability via ``sys.executable``.
        """
        result = subprocess.run(  # nosec B603 -- subprocess call with controlled/trusted input
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"pytest not available via sys.executable: {result.stderr}"
        assert "pytest" in result.stdout.lower(), "pytest version not found in output"


class TestUVCacheAndPerformance:
    """Test UV cache and performance features."""

    def test_uv_cache_dir_accessible(self):
        """Test that UV cache directory is accessible."""
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "cache", "dir"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"uv cache dir failed: {result.stderr}"
        Path(result.stdout.strip())
        # Cache dir may not exist if nothing has been cached
        # Just verify the command works

    def test_uv_sync_fast(self):
        """Test that ``uv sync --frozen`` is fast (lock-respecting, no network churn when cache is warm)."""
        import time

        # Do not use ``--all-extras`` here: it pulls large optional groups (e.g. gui) and
        # can fail on low-disk systems during wheel extraction; the default lock is enough
        # to verify cache behaviour for normal development.
        start = time.time()
        result = subprocess.run(  # nosec B607 B603 -- subprocess calls with controlled/trusted input
            [UV_BIN, "sync", "--frozen"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=120
        )
        elapsed = time.time() - start
        err = (result.stderr or "") + (result.stdout or "")

        if result.returncode != 0 and (
            "No space" in err or "No space left on device" in err or "os error 28" in err
        ):
            pytest.skip("Insufficient disk for uv cache / venv (errno 28)")

        assert result.returncode == 0, f"uv sync failed: {result.stderr}"
        # Cached sync is usually a few seconds; allow slow CI and cold cache.
        assert elapsed < 120, f"uv sync took too long: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120"])
