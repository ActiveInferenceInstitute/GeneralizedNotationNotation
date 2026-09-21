#!/usr/bin/env python3
"""Tests for ``gnn.execute.subprocess_envelope.run_subprocess_envelope``.

The envelope is the shared execution contract for every GNN execution
backend; these tests pin its failure-mode conversion semantics.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute.subprocess_envelope import (  # noqa: E402
    CancelToken,
    run_subprocess_envelope,
)

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
    "missing_key",
    ["success", "return_code", "stdout", "stderr", "duration_seconds", "cancelled"],
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


def test_input_support_pipes_stdin_to_child() -> None:
    """Wave-2 MIN-03: the envelope can feed stdin (no raw bypass needed)."""
    envelope = run_subprocess_envelope(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
        timeout=30,
        input="envelope-stdin-ok",
    )
    assert envelope["success"] is True
    assert envelope["return_code"] == 0
    assert "envelope-stdin-ok" in envelope["stdout"]


def test_sandbox_false_runs_unsandboxed_with_receipt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """sandbox=False runs the command verbatim and emits the disabled receipt."""
    import logging

    with caplog.at_level(logging.WARNING, logger="gnn.execute.subprocess_envelope"):
        result = run_subprocess_envelope(
            [PYTHON, "-c", "print('unsandboxed-ok')"], sandbox=False
        )
    assert result["success"] is True
    assert result["sandbox_mode"] == "off"
    assert result["sandboxed"] is False
    assert "unsandboxed-ok" in result["stdout"]
    receipts = [
        r
        for r in caplog.records
        if r.__dict__.get("event") == "sandbox_disabled_receipt"
    ]
    assert receipts, "sandbox=False must emit sandbox_disabled_receipt"


def test_sandbox_default_off_mode_runs_unsandboxed_with_receipt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default sandbox=True with GNN_SANDBOX=off runs unsandboxed with receipt."""
    import logging

    with caplog.at_level(logging.WARNING, logger="gnn.execute.subprocess_envelope"):
        result = run_subprocess_envelope([PYTHON, "-c", "print('default-off-ok')"])
    assert result["success"] is True
    assert result["sandbox_mode"] == "off"
    assert result["sandboxed"] is False
    receipts = [
        r
        for r in caplog.records
        if r.__dict__.get("event") == "sandbox_disabled_receipt"
    ]
    assert receipts, "GNN_SANDBOX=off default must emit sandbox_disabled_receipt"


def test_sandbox_require_without_backend_refuses_to_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GNN_SANDBOX=require with no backend blocks before any subprocess runs."""
    import subprocess as sp

    from gnn.execute import subprocess_envelope as se

    monkeypatch.setenv("GNN_SANDBOX", "require")
    monkeypatch.setattr(
        "gnn.execute.sandbox.detect_sandbox", lambda: None, raising=False
    )

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("subprocess must not run when sandbox require is blocked")

    monkeypatch.setattr(sp, "run", _boom)
    result = run_subprocess_envelope([PYTHON, "-c", "print('must-not-run')"])
    assert result["success"] is False
    assert result["return_code"] == se.NEVER_STARTED
    assert result["error_type"] == "SandboxUnavailable"
    assert "require" in result["error"]


def test_sandbox_prefer_wraps_command_when_backend_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GNN_SANDBOX=prefer with a backend prefixes the command vector."""
    from gnn.execute.sandbox import SandboxSpec

    monkeypatch.setenv("GNN_SANDBOX", "prefer")
    monkeypatch.setattr(
        "gnn.execute.sandbox.detect_sandbox",
        lambda: SandboxSpec("echo", ("echo", "SANDBOXED")),
    )
    result = run_subprocess_envelope([PYTHON, "-c", "print('wrapped')"])
    assert result["success"] is True
    assert result["sandboxed"] is True
    assert result["sandbox_mode"] == "prefer"
    assert "SANDBOXED" in result["stdout"]


def test_cancelled_is_false_on_non_cancel_paths() -> None:
    """``cancelled`` is additive: False on success, timeout, and spawn failure."""
    ok = run_subprocess_envelope([PYTHON, "-c", "pass"])
    timed_out = run_subprocess_envelope(
        [PYTHON, "-c", "import time; time.sleep(30)"], timeout=1
    )
    spawn_failure = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    for envelope in (ok, timed_out, spawn_failure):
        assert envelope["cancelled"] is False


def test_cancel_before_spawn_spawns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token already cancelled at entry reports cancelled without spawning."""
    import subprocess as sp

    from gnn.execute import subprocess_envelope as se

    token = CancelToken()
    token.cancel("pre-spawn-abort")
    spawn_attempts: list[object] = []

    def _refuse_to_spawn(*args: object, **kwargs: object) -> object:
        spawn_attempts.append(args)
        raise AssertionError("Popen must not run for a pre-spawn cancel")

    monkeypatch.setattr(sp, "Popen", _refuse_to_spawn)

    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('must-not-run')"], timeout=30, cancel_token=token
    )

    assert spawn_attempts == []
    assert result["cancelled"] is True
    assert result["success"] is False
    assert result["error_type"] == "Cancelled"
    assert result["return_code"] == se.NEVER_STARTED
    assert result["error"] == "Execution cancelled: pre-spawn-abort"
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_cancel_mid_flight_kills_group_and_preserves_partial_output() -> None:
    """A cancel from a second thread beats a long explicit timeout: the poll
    loop observes the token within one slice, group-kills, and drains the
    partial streams instead of waiting out the 30s deadline."""
    token = CancelToken()

    def _cancel_soon() -> None:
        time.sleep(0.5)
        token.cancel("user-abort")

    canceller = threading.Thread(target=_cancel_soon, daemon=True)
    canceller.start()
    result = run_subprocess_envelope(
        [
            PYTHON,
            "-c",
            "print('partial-line', flush=True); import time; time.sleep(30)",
        ],
        timeout=30,
        cancel_token=token,
    )
    canceller.join(timeout=2)

    assert result["cancelled"] is True
    assert result["success"] is False
    assert result["error_type"] == "Cancelled"
    assert result["return_code"] == -1
    assert result["error"] == "Execution cancelled: user-abort"
    assert "partial-line" in result["stdout"]
    assert result["duration_seconds"] < 10.0  # cancelled at ~0.5s, not 30s


def test_timeout_kills_whole_process_group_including_grandchildren(
    tmp_path: Path,
) -> None:
    """Regression: the timeout kill must reap the whole spawned tree.

    The child spawns a grandchild that writes a sentinel ~3s in; the parent
    is killed at timeout=1. A child-only kill orphans the grandchild, which
    then writes the sentinel — this assertion fails on that regression.
    """
    sentinel = tmp_path / "sentinel"
    grandchild_code = (
        "import pathlib, time\n"
        f"time.sleep(3)\npathlib.Path({str(sentinel)!r}).write_text('x')\n"
    )
    child_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
        "time.sleep(30)\n"
    )
    result = run_subprocess_envelope([PYTHON, "-c", child_code], timeout=1)

    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"
    time.sleep(3.5)  # the grandchild would write the sentinel at ~3s if orphaned
    assert not sentinel.exists(), "grandchild survived the timeout group-kill"
