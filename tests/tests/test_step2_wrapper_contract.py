"""Contract for the Step 2 wrapper surface: ``tests.run_tests`` routing args."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast


def test_run_tests_signature_matches_documented_contract() -> None:
    """2_tests.py and docs/gnn/modules/02_tests.md depend on this signature.

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
    source = (
        Path(__file__).resolve().parents[2] / "src" / "gnn" / "2_tests.py"
    ).read_text(encoding="utf-8")
    assert "from tests import run_tests" in source
    assert "SKIP_TESTS_IN_PIPELINE" in source


def test_step2_script_mode_resolves_tests_package(tmp_path: Path) -> None:
    """Script-mode ``python src/gnn/2_tests.py`` must resolve the repo-root
    ``tests`` package without PYTHONPATH crutches (regression: tests/ moved
    out of src/ in the v3.3.0 reorg while the step only bootstrapped src/)."""
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "SKIP_TESTS_IN_PIPELINE": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "src" / "gnn" / "2_tests.py"),
            "--target-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tests will be skipped" in proc.stdout + proc.stderr
