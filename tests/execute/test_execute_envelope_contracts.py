#!/usr/bin/env python3
"""Subprocess-envelope contract pins for execution surfaces without coverage.

Pins CURRENT behavior of the already-migrated shared-envelope callers that
no other test file exercises with a spy:

- ``GNNExecutor`` dispatch defaults (``executor.py``): per-type command
  vectors and timeout fallbacks, plus envelope passthrough.
- ``activeinference_runner``: script execution call shape
  (JULIA_PROJECT env, timeout 600, cwd) and result mapping, plus the
  package-validation probe (timeout 30).
- ``stan_runner.execute_stan_script``: command/env/cwd/timeout and the
  result-dict mapping.
- ``rxinfer_runner`` TOML branch: the ``rxinfer_runner.jl`` command shape.

No Julia, Stan toolchain, or real subprocess is needed — the envelope is
spied and returns canned results (same pattern as ``test_execute_bnlearn``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gnn.execute.activeinference_jl import activeinference_runner
from gnn.execute.executor import GNNExecutor
from gnn.execute.rxinfer import rxinfer_runner
from gnn.execute.stan.stan_runner import execute_stan_script
from gnn.execute.rxinfer.rxinfer_runner import execute_rxinfer_script


def _envelope(**overrides: Any) -> Dict[str, Any]:
    """Full canned envelope, including fields only failures populate."""
    envelope: Dict[str, Any] = {
        "success": True,
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "duration_seconds": 0.01,
        "sandbox_mode": "off",
        "sandboxed": False,
        "error": None,
        "error_type": None,
    }
    envelope.update(overrides)
    return envelope


class _EnvelopeSpy:
    """Capture ``run_subprocess_envelope`` calls; return queued envelopes."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.results = list(results or [])

    def __call__(self, command: List[str], **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"command": list(command), "kwargs": kwargs})
        queued = self.results.pop(0) if self.results else _envelope()
        return dict(queued)


def _spy(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    results: Optional[List[Dict[str, Any]]] = None,
) -> _EnvelopeSpy:
    spy = _EnvelopeSpy(results)
    monkeypatch.setattr(target, spy)
    return spy


# ── GNNExecutor dispatch defaults ──────────────────────────────────────────


def test_pymdp_dispatch_runs_python_with_default_timeout_600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m.py"
    script.write_text("print('ok')\n")
    canned = _envelope(stdout="sim done")
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    result = GNNExecutor()._execute_pymdp_script(str(script))

    assert spy.calls[0]["command"] == [sys.executable, str(script)]
    assert spy.calls[0]["kwargs"]["timeout"] == 600
    # The dispatch methods return the envelope untouched — every field
    # (including sandbox bookkeeping and error slots) passes through.
    assert result == canned


def test_rxinfer_dispatch_runs_julia_on_config_with_default_timeout_300(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "m_config.toml"
    config.write_text("[model]\n")
    canned = _envelope()
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    result = GNNExecutor()._execute_rxinfer_config(str(config))

    assert spy.calls[0]["command"] == ["julia", str(config)]
    assert spy.calls[0]["kwargs"]["timeout"] == 300
    assert result == canned


@pytest.mark.parametrize("execution_type", ["discopy", "jax"])
def test_discopy_and_jax_dispatch_run_sys_executable_with_default_timeout_300(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution_type: str
) -> None:
    target = tmp_path / f"m_{execution_type}.py"
    target.write_text("print('ok')\n")
    canned = _envelope()
    spy = _spy(monkeypatch, "gnn.execute.executor.run_subprocess_envelope", [canned])

    executor = GNNExecutor()
    if execution_type == "discopy":
        result = executor._execute_discopy_diagram(str(target))
    else:
        result = executor._execute_jax_script(str(target))

    assert spy.calls[0]["command"] == [sys.executable, str(target)]
    assert spy.calls[0]["kwargs"]["timeout"] == 300
    assert result == canned


# ── activeinference_runner ─────────────────────────────────────────────────


def _activeinference_script(tmp_path: Path) -> tuple[Path, Path]:
    """Script under an ``activeinference_jl`` dir; returns (script, project_dir)."""
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    script = project_dir / "m_activeinference.jl"
    script.write_text('println("ok")\n')
    return script, project_dir


def test_activeinference_script_success_call_shape_and_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, project_dir = _activeinference_script(tmp_path)
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner."
        "run_subprocess_envelope",
        [_envelope(stdout="ran")],
    )

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False
    )

    assert ok is True
    assert spy.calls[0]["command"] == [
        "julia",
        f"--project={project_dir}",
        str(script.resolve()),
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 600
    assert spy.calls[0]["kwargs"]["env"] == {"JULIA_PROJECT": str(project_dir)}
    assert spy.calls[0]["kwargs"]["cwd"] == str(project_dir)


def test_activeinference_script_appends_output_dir_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, project_dir = _activeinference_script(tmp_path)
    out_dir = tmp_path / "results"
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner."
        "run_subprocess_envelope",
    )

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False, output_dir=out_dir
    )

    assert ok is True
    assert spy.calls[0]["command"][:3] == [
        "julia",
        f"--project={project_dir}",
        str(script.resolve()),
    ]
    assert spy.calls[0]["command"][-2:] == ["--output-dir", str(out_dir)]
    assert out_dir.is_dir()


@pytest.mark.parametrize(
    ("canned", "expected"),
    [
        (_envelope(success=False, return_code=1, stderr="ERROR: package missing"), False),
        (_envelope(success=False, return_code=-1, error="spawn failed"), False),
        (
            _envelope(
                success=False,
                return_code=-1,
                error="killed",
                error_type="TimeoutExpired",
            ),
            False,
        ),
    ],
)
def test_activeinference_script_failure_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canned: Dict[str, Any],
    expected: bool,
) -> None:
    script, _ = _activeinference_script(tmp_path)
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner."
        "run_subprocess_envelope",
        [canned],
    )

    ok = activeinference_runner.execute_activeinference_script(
        script, setup_environment=False
    )

    assert ok is expected
    assert len(spy.calls) == 1


def test_activeinference_package_probe_uses_timeout_30(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    spy = _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner."
        "run_subprocess_envelope",
    )

    ok = activeinference_runner._validate_package(project_dir, "ActiveInference")

    assert ok is True
    assert spy.calls[0]["command"] == [
        "julia",
        f"--project={project_dir}",
        "-e",
        'using ActiveInference; println("✅ ActiveInference loaded")',
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 30
    assert spy.calls[0]["kwargs"]["cwd"] == project_dir  # passed as Path


def test_activeinference_package_probe_failure_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "activeinference_jl"
    project_dir.mkdir()
    _spy(
        monkeypatch,
        "gnn.execute.activeinference_jl.activeinference_runner."
        "run_subprocess_envelope",
        [_envelope(success=False, return_code=1)],
    )

    assert activeinference_runner._validate_package(project_dir, "Missing") is False


# ── stan_runner.execute_stan_script ────────────────────────────────────────


def test_stan_script_default_call_shape_and_result_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m_stan.py"
    script.write_text("print('ok')\n")
    out_dir = tmp_path / "stan_out"
    spy = _spy(
        monkeypatch,
        "gnn.execute.stan.stan_runner.run_subprocess_envelope",
        [_envelope(stdout="sampled")],
    )

    result = execute_stan_script(script, out_dir)

    assert out_dir.is_dir()  # created up front so the child can write into it
    assert spy.calls[0]["command"] == [sys.executable, str(script)]
    assert spy.calls[0]["kwargs"]["timeout"] == 1800
    assert spy.calls[0]["kwargs"]["env"] == {"STAN_OUTPUT_DIR": str(out_dir)}
    assert spy.calls[0]["kwargs"]["cwd"] == str(out_dir)
    assert result == {
        "script": str(script),
        "framework": "stan",
        "return_code": 0,
        "success": True,
        "stdout": "sampled",
        "stderr": "",
        "execution_time_seconds": round(0.01, 3),
        "results_file": str(out_dir / "simulation_results.json"),
    }


def test_stan_script_failure_maps_envelope_and_honors_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "m_stan.py"
    script.write_text("print('ok')\n")
    out_dir = tmp_path / "stan_out"
    spy = _spy(
        monkeypatch,
        "gnn.execute.stan.stan_runner.run_subprocess_envelope",
        [_envelope(success=False, return_code=2, stderr="compile failed")],
    )

    result = execute_stan_script(
        script, out_dir, timeout=60, python_executable="custompython"
    )

    assert spy.calls[0]["command"] == ["custompython", str(script)]
    assert spy.calls[0]["kwargs"]["timeout"] == 60
    assert result["success"] is False
    assert result["return_code"] == 2
    assert result["stderr"] == "compile failed"


# ── rxinfer_runner TOML branch ─────────────────────────────────────────────


def test_rxinfer_toml_config_runs_committed_runner_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "model_a_config.toml"
    config.write_text("[model]\n")
    runner_script = Path(rxinfer_runner.__file__).parent / "rxinfer_runner.jl"
    spy = _spy(
        monkeypatch,
        "gnn.execute.rxinfer.rxinfer_runner.run_subprocess_envelope",
        [_envelope()],
    )

    ok = execute_rxinfer_script(config, output_dir=tmp_path / "logs")

    assert ok is True
    assert spy.calls[0]["command"] == [
        "julia",
        "--startup-file=no",
        f"--project={Path(rxinfer_runner.__file__).parent.resolve()}",
        str(runner_script),
        str(config),
    ]
    assert spy.calls[0]["kwargs"]["timeout"] == 300


def test_rxinfer_toml_config_missing_runner_script_refuses_without_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without rxinfer_runner.jl beside the module, TOML configs refuse cleanly."""
    config = tmp_path / "model_a_config.toml"
    config.write_text("[model]\n")
    spy = _spy(
        monkeypatch, "gnn.execute.rxinfer.rxinfer_runner.run_subprocess_envelope"
    )

    calls: Dict[str, Any] = {}
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "rxinfer_runner.jl":
            calls["queried"] = True
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert execute_rxinfer_script(config, output_dir=tmp_path / "logs") is False
    assert spy.calls == []
    assert calls.get("queried") is True
