"""execute_rxinfer_script: execution-evidence persistence contract.

Mirrors the jax_runner pattern: every completed run (success, failure, or
timeout) writes ``{stem}_stdout.txt``, ``{stem}_stderr.txt``, and
``{stem}_execution_log.json``. No Julia is needed — the subprocess envelope
is faked; these tests pin persistence, not Julia behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from gnn.execute.rxinfer.rxinfer_runner import execute_rxinfer_script


@pytest.fixture()
def script_file(tmp_path: Path) -> Path:
    script = tmp_path / "model_a_rxinfer.jl"
    script.write_text('println("RxInfer simulation complete")\n')
    return script


def _stub_runner(
    monkeypatch: pytest.MonkeyPatch, envelope: Dict[str, Any]
) -> Dict[str, Any]:
    """Replace the subprocess invocation with a canned envelope; record calls."""
    calls: Dict[str, Any] = {"count": 0, "command": None}

    def fake_run(cmd: list[str], timeout: Any = None) -> Dict[str, Any]:
        calls["count"] += 1
        calls["command"] = cmd
        return dict(envelope)

    monkeypatch.setattr(
        "gnn.execute.rxinfer.rxinfer_runner.run_subprocess_envelope", fake_run
    )
    return calls


def test_failing_run_writes_all_three_evidence_files(
    tmp_path: Path, script_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "logs"
    calls = _stub_runner(
        monkeypatch,
        {
            "success": False,
            "return_code": 1,
            "stdout": "partial progress output\n",
            "stderr": "ERROR: LoadError: model failed\n",
            "duration_seconds": 1.5,
        },
    )

    result = execute_rxinfer_script(script_file, verbose=False, output_dir=out_dir)

    assert result is False
    assert calls["count"] == 1

    stem = script_file.stem
    assert (out_dir / f"{stem}_stdout.txt").read_text() == "partial progress output\n"
    assert (out_dir / f"{stem}_stderr.txt").read_text() == (
        "ERROR: LoadError: model failed\n"
    )

    log = json.loads((out_dir / f"{stem}_execution_log.json").read_text())
    assert log["success"] is False
    assert log["return_code"] == 1
    assert log["command"][:2] == ["julia", "--startup-file=no"]
    assert log["command"][-1] == str(script_file)
    assert log["elapsed_seconds"] == 1.5
    assert log["timeout"] == 300
    assert "timestamp" in log


def test_successful_run_writes_all_three_evidence_files(
    tmp_path: Path, script_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "logs"
    _stub_runner(
        monkeypatch,
        {
            "success": True,
            "return_code": 0,
            "stdout": "RxInfer simulation complete\n",
            "stderr": "",
            "duration_seconds": 0.25,
        },
    )

    result = execute_rxinfer_script(script_file, verbose=True, output_dir=out_dir)

    assert result is True
    stem = script_file.stem
    assert (out_dir / f"{stem}_stdout.txt").read_text() == (
        "RxInfer simulation complete\n"
    )
    assert (out_dir / f"{stem}_stderr.txt").read_text() == ""

    log = json.loads((out_dir / f"{stem}_execution_log.json").read_text())
    assert log["success"] is True
    assert log["return_code"] == 0
    assert log["elapsed_seconds"] == 0.25


def test_timeout_run_still_writes_evidence(
    tmp_path: Path, script_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out run persists partial output and is reported as failure."""
    out_dir = tmp_path / "logs"
    _stub_runner(
        monkeypatch,
        {
            "success": False,
            "return_code": -1,
            "stdout": "killed mid-simulation output\n",
            "stderr": "drained partial stderr\n",
            "duration_seconds": 5.0,
            "error": "Execution timed out after 5s",
            "error_type": "TimeoutExpired",
        },
    )

    result = execute_rxinfer_script(script_file, timeout=5, output_dir=out_dir)

    assert result is False
    stem = script_file.stem
    assert (out_dir / f"{stem}_stdout.txt").read_text() == (
        "killed mid-simulation output\n"
    )
    assert (out_dir / f"{stem}_stderr.txt").read_text() == "drained partial stderr\n"

    log = json.loads((out_dir / f"{stem}_execution_log.json").read_text())
    assert log["return_code"] == -1
    assert log["success"] is False
    assert log["error_type"] == "TimeoutExpired"
    assert log["error"] == "Execution timed out after 5s"
    assert log["timeout"] == 5


def test_default_log_dir_is_the_script_directory(
    tmp_path: Path, script_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no output_dir, evidence lands beside the script (jax fallback)."""
    _stub_runner(
        monkeypatch,
        {
            "success": False,
            "return_code": 2,
            "stdout": "",
            "stderr": "boom\n",
            "duration_seconds": 0.1,
        },
    )

    assert execute_rxinfer_script(script_file, verbose=False) is False

    stem = script_file.stem
    assert (tmp_path / f"{stem}_stdout.txt").exists()
    assert (tmp_path / f"{stem}_stderr.txt").exists()
    log = json.loads((tmp_path / f"{stem}_execution_log.json").read_text())
    assert log["script"] == str(script_file.resolve())


def test_explicit_output_dir_overrides_script_directory(
    tmp_path: Path, script_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "elsewhere"
    _stub_runner(
        monkeypatch,
        {
            "success": True,
            "return_code": 0,
            "stdout": "ok\n",
            "stderr": "",
            "duration_seconds": 0.1,
        },
    )

    assert execute_rxinfer_script(script_file, output_dir=out_dir) is True

    stem = script_file.stem
    assert (out_dir / f"{stem}_stdout.txt").read_text() == "ok\n"
    assert (out_dir / f"{stem}_stderr.txt").exists()
    assert (out_dir / f"{stem}_execution_log.json").exists()
    # Nothing beside the script itself.
    assert not (tmp_path / f"{stem}_stdout.txt").exists()


def test_persistence_failure_does_not_mask_run_result(
    tmp_path: Path, script_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort evidence writing: an unwritable log dir must not flip the verdict."""
    _stub_runner(
        monkeypatch,
        {
            "success": True,
            "return_code": 0,
            "stdout": "ok\n",
            "stderr": "",
            "duration_seconds": 0.1,
        },
    )

    # An existing FILE at the log-dir path makes mkdir(parents=True,
    # exist_ok=True) raise, exercising the best-effort guard end to end.
    blocked_dir = tmp_path / "occupied"
    blocked_dir.write_text("a file, not a directory\n")

    assert execute_rxinfer_script(script_file, output_dir=blocked_dir) is True
