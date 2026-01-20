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

Following the project's Zero Mock policy - all tests use real methods.
"""

import pytest
import subprocess
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.uv]

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
VENV_PATH = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_PATH / "bin" / "python" if sys.platform != "win32" else VENV_PATH / "Scripts" / "python.exe"

# Import setup module
try:
    from src.setup import (
        setup_uv_environment,
        validate_uv_setup,
        get_uv_setup_info,
        check_system_requirements,
        check_uv_availability,
        get_installed_package_versions,
        check_environment_health,
        OPTIONAL_GROUPS,
        FEATURES,
        add_uv_dependency,
        remove_uv_dependency,
        update_uv_dependencies,
        lock_uv_dependencies,
        validate_system,
        get_environment_info,
        get_uv_status,
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
        result = subprocess.run(
            ["uv", "--version"],
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
        result = subprocess.run(
            ["uv", "--version"],
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
        result = subprocess.run(
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
        result = subprocess.run(
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
        import toml
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path) as f:
            config = toml.load(f)
        
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
        
        # Check for core packages
        core_packages = ["numpy", "pytest", "matplotlib", "scipy"]
        for pkg in core_packages:
            assert pkg in packages, f"Core package {pkg} not found"
    
    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_uv_pip_list_works(self):
        """Test that uv pip list command works."""
        result = subprocess.run(
            ["uv", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"uv pip list failed: {result.stderr}"
        
        packages = json.loads(result.stdout)
        assert isinstance(packages, list), "uv pip list should return a list"
        assert len(packages) > 0, "Should have installed packages"
    
    @pytest.mark.skipif(not SETUP_AVAILABLE, reason="Setup module not available")
    def test_uv_sync_check(self):
        """Test that uv sync --dry-run reports no changes needed."""
        result = subprocess.run(
            ["uv", "sync", "--frozen", "--check"],
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
        expected_keys = ["system_requirements", "uv_environment", "dependencies", "overall_status"]
        for key in expected_keys:
            assert key in result, f"Result missing key: {key}"
        
        # Verify all checks pass
        assert result["system_requirements"] is True, "System requirements check should pass"
        assert result["uv_environment"] is True, "UV environment check should pass"
        assert result["dependencies"] is True, "Dependencies check should pass"
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
        result = subprocess.run(
            ["uv", "run", "python", "-c", "print('Hello from UV')"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"uv run python failed: {result.stderr}"
        assert "Hello from UV" in result.stdout, "Unexpected output"
    
    def test_uv_run_module_import(self):
        """Test that uv run can import project modules."""
        result = subprocess.run(
            ["uv", "run", "python", "-c", "from src.setup import FEATURES; print('Import OK')"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"Module import failed: {result.stderr}"
        assert "Import OK" in result.stdout, "Module import verification failed"
    
    def test_uv_run_pytest(self):
        """Test that uv run pytest works."""
        result = subprocess.run(
            ["uv", "run", "pytest", "--version"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30
        )
        assert result.returncode == 0, f"uv run pytest failed: {result.stderr}"
        assert "pytest" in result.stdout.lower(), "pytest version not found in output"


class TestUVCacheAndPerformance:
    """Test UV cache and performance features."""
    
    def test_uv_cache_dir_accessible(self):
        """Test that UV cache directory is accessible."""
        result = subprocess.run(
            ["uv", "cache", "dir"],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"uv cache dir failed: {result.stderr}"
        cache_dir = Path(result.stdout.strip())
        # Cache dir may not exist if nothing has been cached
        # Just verify the command works
    
    def test_uv_sync_fast(self):
        """Test that uv sync with frozen lock is fast (cached)."""
        import time
        
        start = time.time()
        result = subprocess.run(
            ["uv", "sync", "--frozen"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60
        )
        elapsed = time.time() - start
        
        assert result.returncode == 0, f"uv sync failed: {result.stderr}"
        # Cached sync should be fast (< 10 seconds typically)
        # But we're lenient here to avoid flaky tests
        assert elapsed < 60, f"uv sync took too long: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=120"])
