#!/usr/bin/env python3
"""Tests for ``gnn.execute.subprocess_envelope.run_subprocess_envelope``.

The envelope is the shared execution contract for every GNN execution
backend; these tests pin its failure-mode conversion semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute.subprocess_envelope import run_subprocess_envelope  # noqa: E402

PYTHON = sys.executable


def test_success_envelope() -> None:
    result = run_subprocess_envelope([PYTHON, "-c", "print('hello-from-envelope')"])
    assert result["success"] is True
    assert result["return_code"] == 0
    assert "hello-from-envelope" in result["stdout"]
    assert result["stderr"] == ""
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] >= 0.0
    assert "error" not in result


def test_nonzero_exit_converted() -> None:
    result = run_subprocess_envelope([PYTHON, "-c", "raise SystemExit(3)"])
    assert result["success"] is False
    assert result["return_code"] == 3
    assert "error" not in result


def test_timeout_converted() -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import time; time.sleep(30)"], timeout=1
    )
    assert result["success"] is False
    assert result["return_code"] == -1
    assert result["error_type"] == "TimeoutExpired"
    assert "timed out after 1s" in result["error"]


def test_oserror_converted() -> None:
    result = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    assert result["success"] is False
    assert result["return_code"] == -1
    assert result["error_type"] == "FileNotFoundError"
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_env_overrides_merge_over_parent() -> None:
    result = run_subprocess_envelope(
        [
            PYTHON,
            "-c",
            "import os; print(os.environ.get('GNN_ENVELOPE_PROBE', 'missing'))",
        ],
        env={"GNN_ENVELOPE_PROBE": "present"},
    )
    assert result["success"] is True
    assert "present" in result["stdout"]


def test_capture_output_false_yields_empty_streams() -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('streamed')"], capture_output=False
    )
    assert result["success"] is True
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_cwd_is_honored(tmp_path: Path) -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import os; print(os.getcwd())"], cwd=str(tmp_path)
    )
    assert result["success"] is True
    assert str(tmp_path) in result["stdout"]


def test_command_is_argument_vector_not_shell(tmp_path: Path) -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('no shell interpolation $HOME')"]
    )
    assert result["success"] is True


@pytest.mark.parametrize(
    "missing_key", ["success", "return_code", "stdout", "stderr", "duration_seconds"]
)
def test_envelope_always_carries_core_keys(missing_key: str) -> None:
    ok = run_subprocess_envelope([PYTHON, "-c", "pass"])
    bad = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    assert missing_key in ok
    assert missing_key in bad


def test_timeout_captured_streams_are_str_with_partial_output() -> None:
    """TimeoutExpired streams land as documented ``str`` (CPython delivers
    bytes under ``text=True``); partial pre-kill output is preserved."""
    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('partial-line', flush=True); import time; time.sleep(5)"],
        timeout=1,
    )
    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"
    assert "timed out after 1s" in result["error"]
    assert result["stdout"] == "partial-line\n"
    assert result["stderr"] == ""


def test_timeout_without_capture_yields_empty_streams() -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import time; time.sleep(5)"],
        timeout=1,
        capture_output=False,
    )
    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"
    assert result["stdout"] == ""
    assert result["stderr"] == ""
