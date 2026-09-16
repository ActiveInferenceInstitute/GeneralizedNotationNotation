#!/usr/bin/env python3
"""bnlearn executor: language-aware discovery, probes, and skip semantics.

All tests are offline — subprocess envelopes and runtime probes are
monkeypatched, so neither the Python ``bnlearn`` module nor R/Rscript is
required. The Step 12 wiring points (execution environment, planner
disposition, pre-flight skip) are exercised directly.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.execute import planning as execute_planning  # noqa: E402
from gnn.execute import processor as execute_processor  # noqa: E402
from gnn.execute.bnlearn import bnlearn_runner  # noqa: E402
from gnn.execute.bnlearn.bnlearn_runner import (  # noqa: E402
    OUTPUT_ENV_VAR,
    execute_bnlearn_script,
    find_bnlearn_scripts,
    is_bnlearn_available,
    is_r_bnlearn_available,
    run_bnlearn_scripts,
    script_language,
)
from gnn.execute.types import ScriptExecutionContext  # noqa: E402
from gnn.utils.runtime_safety.framework_availability import (  # noqa: E402
    FRAMEWORK_IMPORT_CHECK,
)


def _write_render_script(
    root: Path, model: str = "m", name: str = "m_bnlearn.py"
) -> Path:
    """Create a rendered script under ``<root>/<model>/bnlearn/``."""
    script = root / model / "bnlearn" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n")
    return script


def _envelope(**overrides: Any) -> Dict[str, Any]:
    envelope: Dict[str, Any] = {
        "success": True,
        "return_code": 0,
        "stdout": "ok",
        "stderr": "",
        "duration_seconds": 0.01,
        "sandbox_mode": "off",
        "sandboxed": False,
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


# ── Discovery and language detection ───────────────────────────────────────


def test_find_bnlearn_scripts_matches_only_bnlearn_render_dirs(
    tmp_path: Path,
) -> None:
    py_script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    r_script = _write_render_script(tmp_path, "model_a", "b.R")
    (tmp_path / "model_a" / "jax").mkdir()
    (tmp_path / "model_a" / "jax" / "other_stan.py").write_text("print()\n")
    (tmp_path / "model_a" / "jax" / "x.py").write_text("print()\n")
    (tmp_path / "model_a" / "pymdp").mkdir()
    (tmp_path / "model_a" / "pymdp" / "y.py").write_text("print()\n")

    assert find_bnlearn_scripts(tmp_path) == sorted([py_script, r_script])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("m_bnlearn.py", "python"),
        ("m.R", "r"),
        ("m.r", "r"),
        ("m.jl", "unknown"),
        ("m", "unknown"),
    ],
)
def test_script_language_derived_from_suffix(name: str, expected: str) -> None:
    assert script_language(Path(name)) == expected


# ── Availability probes ────────────────────────────────────────────────────


def test_bnlearn_import_check_is_registered() -> None:
    """The shared probe maps bnlearn → the bnlearn module + install hint."""
    assert FRAMEWORK_IMPORT_CHECK["bnlearn"] == ("bnlearn", "uv sync")


def test_python_probe_delegates_to_shared_availability(monkeypatch: Any) -> None:
    seen: Dict[str, Any] = {}

    def _fake_probe(
        framework: str,
        executor: Optional[str] = None,
        logger: Any = None,
    ) -> bool:
        seen["framework"] = framework
        seen["executor"] = executor
        return True

    monkeypatch.setattr(bnlearn_runner, "is_framework_available", _fake_probe)
    assert is_bnlearn_available("python3") is True
    assert seen == {"framework": "bnlearn", "executor": "python3"}


def test_r_probe_requires_rscript_on_path(monkeypatch: Any) -> None:
    monkeypatch.setattr(bnlearn_runner.shutil, "which", lambda name: None)
    assert is_r_bnlearn_available() is False


def test_r_probe_probes_library_load(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        bnlearn_runner.shutil, "which", lambda name: "/usr/bin/Rscript"
    )
    spy = _EnvelopeSpy([_envelope()])
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    assert is_r_bnlearn_available() is True
    call = spy.calls[0]
    assert call["command"][0] == "/usr/bin/Rscript"
    assert "library(bnlearn)" in call["command"][2]
    assert call["kwargs"].get("sandbox", True) is False


# ── Direct runner records ──────────────────────────────────────────────────


def test_execute_python_script_success_sets_output_env(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    out_dir = tmp_path / "exec" / "model_a"
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner, "is_bnlearn_available", lambda executor=None: True
    )

    record = execute_bnlearn_script(script, out_dir)

    assert record["success"] is True
    assert record["skipped"] is False
    assert record["framework"] == "bnlearn"
    assert record["language"] == "python"
    assert record["return_code"] == 0
    assert out_dir.is_dir()
    call = spy.calls[0]
    assert call["command"] == [sys.executable, str(script)]
    assert call["kwargs"]["env"] == {OUTPUT_ENV_VAR: str(out_dir)}
    assert call["kwargs"]["cwd"] == str(out_dir)
    # Sandbox semantics delegated to the shared envelope (default enabled).
    assert call["kwargs"].get("sandbox", True) is True


def test_execute_python_script_skips_when_module_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner, "is_bnlearn_available", lambda executor=None: False
    )

    record = execute_bnlearn_script(script, tmp_path / "exec")

    assert record["skipped"] is True
    assert record["success"] is False
    assert "uv sync --extra bnlearn" in record["reason"]
    assert record["return_code"] is None
    assert spy.calls == []


def test_execute_python_script_nonzero_exit_fails_explicitly(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    spy = _EnvelopeSpy(
        [
            _envelope(
                success=False,
                return_code=2,
                stdout="",
                stderr="boom",
                duration_seconds=0.5,
            )
        ]
    )
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner, "is_bnlearn_available", lambda executor=None: True
    )

    record = execute_bnlearn_script(script, tmp_path / "exec")

    assert record["success"] is False
    assert record["skipped"] is False
    assert record["return_code"] == 2
    assert record["stderr"] == "boom"
    assert record["error_type"] == "RuntimeError"
    assert "failed (2)" in record["error"]


def test_execute_python_script_timeout_reports_error_type(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    spy = _EnvelopeSpy(
        [
            _envelope(
                success=False,
                return_code=-1,
                stdout="partial",
                stderr="partial",
                duration_seconds=2.0,
                error_type="TimeoutExpired",
            )
        ]
    )
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner, "is_bnlearn_available", lambda executor=None: True
    )

    record = execute_bnlearn_script(script, tmp_path / "exec", timeout=5)

    assert record["success"] is False
    assert record["skipped"] is False
    assert record["error_type"] == "TimeoutExpired"
    assert "timed out after 5s" in record["error"]


def test_execute_r_script_runs_under_rscript(tmp_path: Path, monkeypatch: Any) -> None:
    script = _write_render_script(tmp_path, "model_a", "a.R")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(bnlearn_runner, "is_r_bnlearn_available", lambda r="Rscript": True)

    record = execute_bnlearn_script(script, tmp_path / "exec", rscript_executable="myR")

    assert record["success"] is True
    assert record["language"] == "r"
    assert spy.calls[0]["command"] == ["myR", str(script)]


def test_execute_unknown_language_skips_without_subprocess(tmp_path: Path) -> None:
    script = _write_render_script(tmp_path, "model_a", "a.jl")

    record = execute_bnlearn_script(script, tmp_path / "exec")

    assert record["skipped"] is True
    assert record["success"] is False
    assert "unknown" in record["language"] or record["language"] == "unknown"
    assert "Unsupported bnlearn script language" in record["reason"]


def test_run_bnlearn_scripts_mixed_lanes(tmp_path: Path, monkeypatch: Any) -> None:
    py_script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    r_script = _write_render_script(tmp_path, "model_b", "b.R")
    spy = _EnvelopeSpy()
    monkeypatch.setattr(bnlearn_runner, "run_subprocess_envelope", spy)
    monkeypatch.setattr(
        bnlearn_runner, "is_bnlearn_available", lambda executor=None: False
    )
    monkeypatch.setattr(bnlearn_runner, "is_r_bnlearn_available", lambda r="Rscript": True)

    records = run_bnlearn_scripts(tmp_path, tmp_path / "exec")

    assert records[0]["script"] == str(py_script)
    assert records[0]["skipped"] is True
    assert records[1]["script"] == str(r_script)
    assert records[1]["success"] is True
    assert spy.calls[0]["command"] == ["Rscript", str(r_script)]
    assert spy.calls[0]["kwargs"]["cwd"] == str(tmp_path / "exec" / "model_b")


# ── Step 12 wiring ─────────────────────────────────────────────────────────


def test_step12_execution_environment_sets_bnlearn_output_dir(tmp_path: Path) -> None:
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    context = ScriptExecutionContext(
        script_path=script,
        script_name=script.name,
        framework="bnlearn",
        model_name="model_a",
        executor=sys.executable,
    )

    env = execute_processor._build_execution_environment(context, tmp_path / "12")

    simulation_data = tmp_path / "12" / "model_a" / "bnlearn" / "simulation_data"
    assert env["BNLEARN_OUTPUT_DIR"] == str(simulation_data)
    assert simulation_data.is_dir()


def test_plan_disposition_marks_bnlearn_dependency_skip(monkeypatch: Any) -> None:
    script_info = {
        "framework": "bnlearn",
        "executor": sys.executable,
        "path": "/render/m/bnlearn/m_bnlearn.py",
        "name": "m_bnlearn.py",
    }
    monkeypatch.setattr(
        execute_planning, "is_framework_available", lambda *a, **k: False
    )
    assert execute_planning._disposition(script_info) == "skip_dependency"
    monkeypatch.setattr(
        execute_planning, "is_framework_available", lambda *a, **k: True
    )
    assert execute_planning._disposition(script_info) == "execute"


def test_step12_preflight_skips_bnlearn_scripts_without_module(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The documented skip story: bnlearn scripts skip at the Python
    pre-flight check when the module is absent."""
    script = _write_render_script(tmp_path, "model_a", "a_bnlearn.py")
    script_info = {
        "path": script,
        "name": script.name,
        "framework": "bnlearn",
        "executor": sys.executable,
        "relative_path": script,
        "size_bytes": script.stat().st_size,
    }
    monkeypatch.setattr(
        execute_processor, "_is_framework_available_by_name", lambda *a, **k: False
    )

    result = execute_processor.execute_single_script(
        script_info, tmp_path / "12", False, execute_processor.logger, 60
    )

    assert result["skipped"] is True
    assert result["success"] is False
    assert result["status"] == "skipped"
    assert result["error_type"] == "DependencyNotInstalled"
    assert "bnlearn" in result["error"]
