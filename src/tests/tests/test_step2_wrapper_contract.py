"""Contract for the Step 2 wrapper surface: ``tests.run_tests`` routing args."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def test_run_tests_signature_matches_documented_contract() -> None:
    """2_tests.py and doc/gnn/modules/02_tests.md depend on this signature.

    Import from ``tests.runner`` (the canonical source): under pytest the
    conftest registers a minimal ``sys.modules['tests']`` alias, so the
    package-level attribute is only reliable outside pytest — see
    ``test_tests_package_imports.py``.
    """
    from tests.runner import run_tests

    params = inspect.signature(run_tests).parameters
    assert list(params) == [
        "logger",
        "output_dir",
        "verbose",
        "fast_only",
        "comprehensive",
        "generate_coverage",
        "auto_fallback",
    ]
    assert params["fast_only"].default is True
    assert params["comprehensive"].default is False
    assert params["auto_fallback"].default is True


def test_step2_wrapper_exists_and_is_thin() -> None:
    """``src/2_tests.py`` must keep delegating to ``tests.run_tests``."""
    source = (Path(__file__).resolve().parents[2] / "2_tests.py").read_text(
        encoding="utf-8"
    )
    assert "from tests import run_tests" in source
    assert "SKIP_TESTS_IN_PIPELINE" in source
