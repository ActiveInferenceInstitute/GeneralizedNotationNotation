"""Pins for ``utils/execution_utils.execute_command_streaming`` (previously untested).

Exercises the real subprocess path with fast, offline commands; asserts the
documented result contract (exit_code / stdout / stderr / status).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gnn.utils.execution_utils import execute_command_streaming

PY = sys.executable


def test_successful_command_streams_and_captures() -> None:
    result = execute_command_streaming(
        [PY, "-c", "print('hello-stream')"], capture_output=True
    )

    assert result["status"] == "SUCCESS"
    assert result["exit_code"] == 0
    assert "hello-stream" in result["stdout"]


def test_failing_command_reports_exit_code_and_stderr() -> None:
    result = execute_command_streaming(
        [PY, "-c", "import sys; print('boom-out'); sys.exit(3)"],
        capture_output=True,
    )

    assert result["status"] == "FAILED"
    assert result["exit_code"] == 3
    assert "boom-out" in result["stdout"]


def test_timeout_kills_process_and_reports_status() -> None:
    result = execute_command_streaming(
        [PY, "-c", "import time; print('started'); time.sleep(30)"],
        timeout=2,
        capture_output=True,
    )

    assert result["status"] == "TIMEOUT"
    # The process tree was terminated, so no exit code can be collected.
    assert result["exit_code"] != 0


def test_env_extension_is_visible_to_the_child() -> None:
    result = execute_command_streaming(
        [PY, "-c", "import os; print(os.environ['W2_PROBE_VALUE'])"],
        env={"W2_PROBE_VALUE": "probe-ok"},
        capture_output=True,
    )

    assert result["status"] == "SUCCESS"
    assert "probe-ok" in result["stdout"]


def test_cwd_is_honored(tmp_path: Path) -> None:
    result = execute_command_streaming(
        [PY, "-c", "import os; print(os.getcwd())"],
        cwd=tmp_path,
        capture_output=True,
    )

    assert result["status"] == "SUCCESS"
    assert str(tmp_path) in result["stdout"]


@pytest.mark.parametrize("capture_output", [True, False])
def test_capture_flag_controls_returned_streams(capture_output: bool) -> None:
    result = execute_command_streaming(
        [PY, "-c", "print('captured?')"], capture_output=capture_output
    )

    assert result["status"] == "SUCCESS"
    if capture_output:
        assert "captured?" in result["stdout"]
    else:
        assert result["stdout"] == ""
