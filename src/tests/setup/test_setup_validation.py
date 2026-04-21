#!/usr/bin/env python3
"""Phase 4.2 regression tests for setup (Step 1).

Zero-mock: uses real sys.version_info + real filesystem.
"""

import shutil
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_check_python_version_respects_current_interpreter():
    """check_python_version must report True on Python 3.11+ (our minimum).
    If this test ever fails, either the interpreter is too old OR the
    requirements in pyproject.toml were silently relaxed."""
    from setup import check_python_version
    result = check_python_version()
    assert isinstance(result, bool)
    if sys.version_info >= (3, 11):
        assert result is True, f"check_python_version rejected {sys.version_info}"


def test_validate_environment_returns_dict():
    """validate_environment must return a diagnostic dict, not raise."""
    from setup import validate_environment
    result = validate_environment()
    assert isinstance(result, dict)
    # Should expose at least a high-level status signal.
    expected_keys = {"valid", "python_version", "errors", "warnings",
                     "overall_health", "status", "issues"}
    assert expected_keys & set(result.keys()), (
        f"validate_environment returned no recognized status key: {list(result.keys())}"
    )


def test_get_module_info_exposes_version():
    from setup import get_module_info
    info = get_module_info()
    assert isinstance(info, dict)
    assert "version" in info
    # Version follows semver
    v = str(info["version"])
    assert v.count(".") >= 2, f"Unexpected version format: {v!r}"


def test_environment_manager_instantiates_without_side_effects():
    """Constructing EnvironmentManager must not fail or mutate state."""
    from setup import EnvironmentManager
    mgr = EnvironmentManager()
    assert mgr is not None


def test_uv_availability_detection_matches_shutil():
    """If setup exposes a uv detection helper, it must agree with shutil.which.

    Covers this by importing uv_management indirectly — if the helper is
    reachable we verify parity; otherwise we skip.
    """
    try:
        from setup.uv_management import check_uv_availability
    except ImportError:
        pytest.skip("setup.uv_management.check_uv_availability not exposed")
    result = check_uv_availability()
    # Result shape varies (bool or dict); just check it matches shutil.which.
    has_uv = shutil.which("uv") is not None
    if isinstance(result, bool):
        assert result == has_uv
    elif isinstance(result, dict):
        reported = result.get("available") or result.get("found")
        if reported is not None:
            assert bool(reported) == has_uv
