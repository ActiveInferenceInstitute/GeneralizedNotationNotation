"""Tests for the per-runner pre-execution security gate (SEC-R2 closure).

The rxinfer, activeinference, and bnlearn per-script runners must consult
``check_script_allowed`` before any subprocess spawn: a blocked verdict
refuses execution without touching the envelope, the operator opt-out
(``GNN_ALLOW_UNSAFE_EXEC``) still permits trusted-local runs, and a benign
script passes the real scanner.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

import gnn.execute.rxinfer.rxinfer_runner as rxinfer_runner

_BLOCKED_VERDICT: dict[str, Any] = {
    "ok": False,
    "overridden": False,
    "blocked": [{"vulnerability_type": "os_system", "line": 1}],
    "error_type": "SecurityGateBlocked",
    "reason": "unit-forced block",
}


def _success_envelope() -> dict[str, Any]:
    """Envelope shape matching a completed successful subprocess run."""
    return {
        "success": True,
        "return_code": 0,
        "stdout": "ok\n",
        "stderr": "",
        "error_type": None,
        "error": None,
        "duration_seconds": 0.01,
    }


def _refusing_spawner(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise AssertionError(
        "run_subprocess_envelope must not be called when the gate blocks"
    )


@pytest.mark.parametrize(
    ("module_name", "function_name", "script_name"),
    (
        (
            "gnn.execute.rxinfer.rxinfer_runner",
            "execute_rxinfer_script",
            "model_rxinfer.jl",
        ),
        (
            "gnn.execute.activeinference_jl.activeinference_runner",
            "execute_activeinference_script",
            "model_activeinference.jl",
        ),
    ),
)
def test_gate_block_refuses_to_spawn(
    module_name: str,
    function_name: str,
    script_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked gate verdict returns a bool failure before any spawn."""
    runner_module = importlib.import_module(module_name)
    monkeypatch.setattr(
        runner_module, "check_script_allowed", lambda path: dict(_BLOCKED_VERDICT)
    )
    monkeypatch.setattr(runner_module, "run_subprocess_envelope", _refusing_spawner)

    script = tmp_path / script_name
    script.write_text("x = 1\n", encoding="utf-8")

    per_script = getattr(runner_module, function_name)
    if function_name == "execute_activeinference_script":
        assert per_script(script, setup_environment=False) is False
    else:
        assert per_script(script) is False


def test_bnlearn_gate_block_returns_blocked_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bnlearn mirrors its failure-record shape with the gate verdict keys."""
    bnlearn_module = importlib.import_module("gnn.execute.bnlearn.bnlearn_runner")
    monkeypatch.setattr(
        bnlearn_module, "check_script_allowed", lambda path: dict(_BLOCKED_VERDICT)
    )
    # The gate runs ahead of the lane probes in the landed ordering; keep the
    # probe from shadowing the verdict regardless of that placement.
    monkeypatch.setattr(bnlearn_module, "is_bnlearn_available", lambda *a, **k: True)
    monkeypatch.setattr(bnlearn_module, "run_subprocess_envelope", _refusing_spawner)

    script = tmp_path / "model_bnlearn.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    record = bnlearn_module.execute_bnlearn_script(script, tmp_path / "out")

    assert record["success"] is False
    assert record["skipped"] is False
    assert record["error_type"] == "SecurityGateBlocked"
    assert "Pre-execution security gate blocked" in record["error"]
    assert record["security_findings"] == _BLOCKED_VERDICT["blocked"]


def test_gate_override_allows_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator opt-out lets a trusted-local script reach the envelope."""
    calls: list[list[Any]] = []

    def recording_envelope(command: list[Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        return _success_envelope()

    monkeypatch.setattr(rxinfer_runner, "run_subprocess_envelope", recording_envelope)
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

    script = tmp_path / "model_rxinfer.jl"
    script.write_text("println(1)\n", encoding="utf-8")

    assert rxinfer_runner.execute_rxinfer_script(script) is True
    assert len(calls) == 1


def test_real_gate_passes_benign_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real scanner allows a benign script; the run reaches the envelope."""
    calls: list[list[Any]] = []

    def recording_envelope(command: list[Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        return _success_envelope()

    monkeypatch.setattr(rxinfer_runner, "run_subprocess_envelope", recording_envelope)

    script = tmp_path / "model_rxinfer.jl"
    script.write_text("x = 1\n", encoding="utf-8")

    assert rxinfer_runner.execute_rxinfer_script(script) is True
    assert len(calls) == 1
