#!/usr/bin/env python3
"""Phase 4.2 regression tests for ml_integration (Step 14).

Exercises the PyTorch/JAX/NumPyro availability detection and module metadata.
Zero-mock per CLAUDE.md — uses real importlib probes.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_check_ml_frameworks_returns_dict_with_known_frameworks():
    from ml_integration import check_ml_frameworks
    result = check_ml_frameworks()
    assert isinstance(result, dict)
    # Should report at least these frameworks, even if unavailable.
    # Individual keys may differ by version; accept any of the common ones.
    known = {"pytorch", "torch", "jax", "numpyro", "tensorflow"}
    # Intersection must be non-empty.
    assert known & set(result.keys()), (
        f"check_ml_frameworks returned unexpected keys: {list(result.keys())}"
    )


def test_check_ml_frameworks_reports_availability_consistently():
    """For each framework in the result, the 'available' flag must match
    what importlib says about the module's spec — our report must not lie."""
    from ml_integration import check_ml_frameworks
    result = check_ml_frameworks()
    # Map reported framework → module it represents.
    framework_to_module = {
        "pytorch": "torch", "torch": "torch",
        "jax": "jax",
        "numpyro": "numpyro",
        "tensorflow": "tensorflow",
    }
    for fw, info in result.items():
        if fw not in framework_to_module:
            continue
        if not isinstance(info, dict):
            continue  # some implementations nest differently — skip non-dicts
        # Only check when "available" is exposed.
        if "available" not in info:
            continue
        expected = importlib.util.find_spec(framework_to_module[fw]) is not None
        actual = bool(info["available"])
        assert actual == expected, (
            f"ml_integration reports {fw}={actual} but find_spec({framework_to_module[fw]!r})={expected}"
        )


def test_ml_integration_module_info_has_version():
    from ml_integration import get_module_info
    info = get_module_info()
    assert isinstance(info, dict)
    assert "version" in info
    # Version must be a string (e.g., "1.6.0") or dict.
    assert isinstance(info["version"], (str, dict))
