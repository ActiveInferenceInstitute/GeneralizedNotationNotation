"""Pinning tests for the pymdp script self-heal path (wave-2 MED-03).

``validate_and_clean_pymdp_script`` must return the path to EXECUTE:

- the original path when the script is already valid;
- a sibling ``.cleaned.py`` path when stray ``}`` syntax errors were
  repaired (the original file must be byte-preserved as the Step-11
  rendered-script audit trail);
- ``None`` when the script is unfixable.

The caller (``execute_pymdp_script_with_outputs``) must execute the
CLEANED path, not the original — otherwise a repaired-but-broken script
fails at runtime with its syntax error instead of running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

from gnn.execute.pymdp.pymdp_runner import (
    execute_pymdp_script_with_outputs,
    validate_and_clean_pymdp_script,
)

_VALID_SCRIPT = '''\
"""A valid PyMDP script."""
result = {"status": "success"}
print("script ran")
'''

_BROKEN_SCRIPT = '''\
"""A PyMDP script with a stray } from the renderer."""
result = {"status": "success"}
}
print("script ran")
'''

_UNFIXABLE_SCRIPT = '''\
def broken(:
    return 1
'''


@pytest.fixture
def script_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestValidateAndCleanReturnPath:
    """The cleaner returns the path to execute, not a bool."""

    @pytest.mark.unit
    def test_valid_script_returns_original_path(
        self, script_dir: Path
    ) -> None:
        path = script_dir / "valid.py"
        path.write_text(_VALID_SCRIPT)
        original_bytes = path.read_bytes()
        result = validate_and_clean_pymdp_script(path)
        assert result == path
        # The original must be byte-preserved (no in-place rewrite).
        assert path.read_bytes() == original_bytes

    @pytest.mark.unit
    def test_broken_script_returns_cleaned_sibling_and_preserves_original(
        self, script_dir: Path
    ) -> None:
        path = script_dir / "broken.py"
        path.write_text(_BROKEN_SCRIPT)
        original_bytes = path.read_bytes()
        result = validate_and_clean_pymdp_script(path)
        assert result is not None
        assert result != path
        assert result.name == "broken.cleaned.py"
        # The original must be byte-preserved (Step-11 audit trail).
        assert path.read_bytes() == original_bytes
        # The cleaned sibling compiles.
        compile(result.read_text(), result.name, "exec")

    @pytest.mark.unit
    def test_unfixable_script_returns_none(self, script_dir: Path) -> None:
        path = script_dir / "unfixable.py"
        path.write_text(_UNFIXABLE_SCRIPT)
        assert validate_and_clean_pymdp_script(path) is None

    @pytest.mark.unit
    def test_missing_script_returns_none(self, script_dir: Path) -> None:
        assert validate_and_clean_pymdp_script(script_dir / "nope.py") is None


class TestSelfHealExecution:
    """The caller must execute the CLEANED path, not the original."""

    @pytest.mark.unit
    def test_broken_script_executes_cleaned_path(
        self, script_dir: Path, tmp_path: Path
    ) -> None:
        """A repaired script must run via the .cleaned.py sibling."""
        script = script_dir / "self_heal.py"
        script.write_text(_BROKEN_SCRIPT)
        original_bytes = script.read_bytes()
        output_dir = tmp_path / "pymdp_outputs"
        output_dir.mkdir()

        result: dict[str, Any] = execute_pymdp_script_with_outputs(
            script, output_dir, verbose=False, timeout=30
        )

        # The original file must be byte-preserved.
        assert script.read_bytes() == original_bytes
        # A .cleaned.py sibling must exist.
        cleaned = script.with_suffix(".cleaned.py")
        assert cleaned.exists()
        # The script must have actually RUN (not failed with a syntax
        # error from the original path). A successful run produces an
        # execution log.
        assert result.get("success") is True, f"result: {result}"
        assert (output_dir / "self_heal" / "self_heal_execution_log.json").exists()
