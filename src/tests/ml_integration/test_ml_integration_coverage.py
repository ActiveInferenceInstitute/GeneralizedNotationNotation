#!/usr/bin/env python3
"""Phase 4.2 regression tests for ml_integration (Step 14).

Exercises the PyTorch/JAX/NumPyro availability detection and module metadata.
Uses real importlib probes.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_check_ml_frameworks_returns_dict_with_known_frameworks() -> Any:
    from ml_integration import check_ml_frameworks

    result = check_ml_frameworks()
    assert isinstance(result, dict)
    # Should report at least these frameworks, even if unavailable.
    # Individual keys may differ by version; accept any of the common ones.
    known: set[Any] = {"pytorch", "torch", "jax", "numpyro", "tensorflow"}
    # Intersection must be non-empty.
    assert known & set(result.keys()), (
        f"check_ml_frameworks returned unexpected keys: {list(result.keys())}"
    )


def test_check_ml_frameworks_reports_availability_consistently() -> Any:
    """For each framework in the result, the 'available' flag must match
    what importlib says about the module's spec — our report must not lie."""
    from ml_integration import check_ml_frameworks

    result = check_ml_frameworks()
    # Map reported framework → module it represents.
    framework_to_module: dict[str, Any] = {
        "pytorch": "torch",
        "torch": "torch",
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


def test_ml_integration_module_info_has_version() -> Any:
    from ml_integration import get_module_info

    info = get_module_info()
    assert isinstance(info, dict)
    assert "version" in info
    # Version must be a string (e.g., "1.6.0") or dict.
    assert isinstance(info["version"], (str, dict))


def test_extract_dimensions_accepts_canonical_greek_identifiers() -> None:
    from ml_integration.processor import _extract_dimensions

    content = "## StateSpaceBlock\nπ[4,type=categorical]\nA[2,3]\n"

    assert _extract_dimensions(content) == {"π": [4], "A": [2, 3]}


def test_recursive_processing_discovers_nested_models(tmp_path: Path) -> None:
    import json

    from ml_integration.processor import process_ml_integration

    nested = tmp_path / "input" / "nested"
    nested.mkdir(parents=True)
    (nested / "model.md").write_text(
        "## StateSpaceBlock\nπ[4,type=categorical]\n", encoding="utf-8"
    )
    output = tmp_path / "output"

    assert process_ml_integration(tmp_path / "input", output, recursive=True)
    results = json.loads((output / "ml_integration_results.json").read_text())
    assert [item["file_name"] for item in results["extracted_features"]] == ["model.md"]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([], 0),
        ([0, 1], 0),
        ([0, 0, 1, 1], 2),
        ([0, 0, 0, 1, 1, 1], 3),
    ],
)
def test_cross_validation_folds_requires_class_support(
    labels: list[int], expected: int
) -> None:
    from ml_integration.processor import _cross_validation_folds

    assert _cross_validation_folds(labels) == expected
