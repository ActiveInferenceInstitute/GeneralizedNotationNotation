"""Regression tests for ``scripts/add_module_docstrings.py``.

The script mutates repository trees, so these tests exercise it against temp
directories only: the dry-run contract (reports candidates, writes nothing)
and the write contract (inserts a docstring as the first statement, skips
already-documented files).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.helpers import load_module_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "add_module_docstrings.py"


def _load() -> Any:
    return load_module_from_path("add_module_docstrings", SCRIPT)


def _run(script_args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *script_args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    """--dry-run lists the candidate file but leaves its content untouched."""
    target = tmp_path / "undocumented.py"
    original = 'def hi():\n    """Say hi."""\n'
    target.write_text(original)
    result = _run([str(tmp_path), "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "undocumented.py" in result.stdout
    assert "would receive" in result.stdout
    assert target.read_text() == original


def test_write_mode_inserts_docstring_and_skips_documented(tmp_path: Path) -> None:
    """Write mode adds a leading docstring and skips already-documented files."""
    module = _load()
    target = tmp_path / "undocumented.py"
    target.write_text('def hi():\n    """Say hi."""\n')
    documented = tmp_path / "documented.py"
    documented.write_text('"""Documented."""\n\nx = 1\n')

    result = module.add_docstring(target)
    assert result is not None
    first_line = target.read_text().splitlines()[0]
    assert first_line.startswith('"""')
    assert "hi" in first_line

    assert module.add_docstring(documented) is None
    assert documented.read_text() == '"""Documented."""\n\nx = 1\n'


def test_script_entrypoint_uses_argparse(tmp_path: Path) -> None:
    """The CLI accepts a root positional plus --dry-run (no bare sys.argv index)."""
    result = _run([str(tmp_path), "--dry-run"])
    assert result.returncode == 0, result.stderr
    assert "0 docstrings" in result.stdout
