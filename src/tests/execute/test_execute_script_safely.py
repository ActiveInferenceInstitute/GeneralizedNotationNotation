"""Tests for ``execute.executor.execute_script_safely``.

Exercises the envelope contract: the function must always return a dict with
``success`` and never raise, regardless of the failure mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from execute import execute_script_safely  # noqa: E402


REQUIRED_KEYS = {
    "success",
    "script_path",
    "return_code",
    "stdout",
    "stderr",
    "duration_seconds",
}


def _assert_envelope(result: dict) -> None:
    assert isinstance(result, dict)
    assert REQUIRED_KEYS.issubset(result.keys()), (
        f"missing keys: {REQUIRED_KEYS - result.keys()}"
    )
    assert isinstance(result["success"], bool)
    assert isinstance(result["return_code"], int)
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert isinstance(result["duration_seconds"], float)


def test_missing_file_returns_failure_envelope(tmp_path: Path) -> None:
    result = execute_script_safely(tmp_path / "does_not_exist.py")
    _assert_envelope(result)
    assert result["success"] is False
    assert result["error_type"] == "FileNotFoundError"


def test_non_python_suffix_rejected(tmp_path: Path) -> None:
    shell_script = tmp_path / "hello.sh"
    shell_script.write_text("echo hello\n")
    result = execute_script_safely(shell_script)
    _assert_envelope(result)
    assert result["success"] is False
    assert result["error_type"] == "ValueError"


def test_successful_script(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hello-from-script')\n")
    result = execute_script_safely(script, timeout=30)
    _assert_envelope(result)
    assert result["success"] is True
    assert result["return_code"] == 0
    assert "hello-from-script" in result["stdout"]


def test_failing_script(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text("import sys; sys.exit(3)\n")
    result = execute_script_safely(script, timeout=30)
    _assert_envelope(result)
    assert result["success"] is False
    assert result["return_code"] == 3


def test_timeout(tmp_path: Path) -> None:
    script = tmp_path / "hang.py"
    script.write_text("import time; time.sleep(5)\n")
    result = execute_script_safely(script, timeout=1)
    _assert_envelope(result)
    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"


def test_env_override_merges(tmp_path: Path) -> None:
    script = tmp_path / "env.py"
    script.write_text(
        "import os; print(os.environ.get('GNN_TEST_VAR', 'MISSING'))\n"
    )
    result = execute_script_safely(
        script, timeout=30, env={"GNN_TEST_VAR": "present"}
    )
    _assert_envelope(result)
    assert result["success"] is True
    assert "present" in result["stdout"]


# --- Phase 1.1 regression: silent-success in process_execute ---------------

def test_process_execute_returns_2_when_no_render_output(tmp_path: Path, monkeypatch) -> None:
    """Before Phase 1.1, process_execute returned True when the render output
    directory was missing — the pipeline reported step 12 as successful while
    silently skipping all work. After the fix it returns 2 ("skipped/warnings")
    so the widened pipeline_template contract can surface this as a warning.

    Isolation note: ``_resolve_render_output_dir`` searches CWD-relative
    ``Path("output")`` as a fallback, so we chdir into an isolated tmp dir
    before invoking the processor; otherwise the real project's render output
    would be discovered and executed.
    """
    from execute.processor import process_execute
    monkeypatch.chdir(tmp_path)  # isolate from real project output/
    empty_target = tmp_path / "no_render_output_here"
    empty_target.mkdir()
    output_dir = tmp_path / "execute_output"
    result = process_execute(
        target_dir=empty_target,
        output_dir=output_dir,
        verbose=False,
        frameworks="all",
    )
    # Per the new contract, "nothing to do" must be exit-code 2, not True.
    assert result == 2, (
        f"Expected exit-code 2 for empty render output; got {result!r}"
    )
