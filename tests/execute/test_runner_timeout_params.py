"""Tests for runner timeout parameters and their propagation.

Every registry batch runner accepts a trailing ``timeout`` keyword; per-script
runners carry their historical defaults (300 s, or the shared
``DEFAULT_RUNNER_TIMEOUT_SECONDS`` for pymdp/activeinference), and the
executor funnel's timeout reaches the real spawn site.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import gnn.execute.activeinference_jl.activeinference_runner as activeinference_runner
import gnn.execute.executor as executor_module
import gnn.execute.pymdp.pymdp_runner as pymdp_runner
import gnn.execute.rxinfer.rxinfer_runner as rxinfer_runner
from gnn.execute.activeinference_jl.activeinference_runner import (
    execute_activeinference_script,
)
from gnn.execute.pymdp.pymdp_runner import run_pymdp_scripts
from gnn.execute.rxinfer.rxinfer_runner import (
    execute_rxinfer_script,
    run_rxinfer_scripts,
)
from gnn.execute.types import DEFAULT_RUNNER_TIMEOUT_SECONDS

# Every registry runner shares one invocation contract: a trailing ``timeout``
# keyword. Batch entries default to None (per-runner historical ceiling);
# per-script entries default to their own concrete value.
_TIMEOUT_CAPABLE_RUNNERS: tuple[tuple[str, str], ...] = (
    ("gnn.execute.rxinfer.rxinfer_runner", "run_rxinfer_scripts"),
    ("gnn.execute.jax.jax_runner", "run_jax_scripts"),
    ("gnn.execute.numpyro.numpyro_runner", "run_numpyro_scripts"),
    ("gnn.execute.pytorch.pytorch_runner", "run_pytorch_scripts"),
    ("gnn.execute.discopy.discopy_executor", "run_discopy_analysis"),
    (
        "gnn.execute.activeinference_jl.activeinference_runner",
        "run_activeinference_analysis",
    ),
    ("gnn.execute.bnlearn.bnlearn_runner", "run_bnlearn_scripts"),
    ("gnn.execute.pymdp.pymdp_runner", "run_pymdp_scripts"),
    (
        "gnn.execute.activeinference_jl.activeinference_runner",
        "execute_activeinference_script",
    ),
    ("gnn.execute.discopy.discopy_executor", "execute_discopy_script"),
)


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


@pytest.mark.parametrize(("module_name", "function_name"), _TIMEOUT_CAPABLE_RUNNERS)
def test_runner_signature_has_timeout(module_name: str, function_name: str) -> None:
    """Every registry runner exposes the uniform ``timeout`` parameter."""
    runner_module = importlib.import_module(module_name)
    signature = inspect.signature(getattr(runner_module, function_name))
    assert "timeout" in signature.parameters


def test_rxinfer_batch_threads_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_rxinfer_scripts`` threads its timeout into every per-script call."""
    captured: list[dict[str, Any]] = []

    def recording_script(script_path: Path, *args: Any, **kwargs: Any) -> bool:
        captured.append(kwargs)
        return True

    # Batch discovery runs a Julia availability gate before scanning; the
    # recorder path under test starts after that gate.
    monkeypatch.setattr(rxinfer_runner, "is_julia_available", lambda: True)
    monkeypatch.setattr(rxinfer_runner, "execute_rxinfer_script", recording_script)

    rxinfer_dir = tmp_path / "rxinfer"
    rxinfer_dir.mkdir()
    (rxinfer_dir / "model_rxinfer.jl").write_text("x = 1\n", encoding="utf-8")

    assert (
        run_rxinfer_scripts(
            tmp_path,
            tmp_path / "out",
            recursive_search=False,
            verbose=False,
            timeout=77,
        )
        is True
    )
    assert len(captured) == 1
    assert captured[0]["timeout"] == 77


def test_rxinfer_script_timeout_reaches_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-script timeout lands on the subprocess envelope call."""
    captured: list[dict[str, Any]] = []

    def recording_envelope(command: list[Any], **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return _success_envelope()

    monkeypatch.setattr(rxinfer_runner, "run_subprocess_envelope", recording_envelope)
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

    script = tmp_path / "m_rxinfer.jl"
    script.write_text("println(1)\n", encoding="utf-8")

    assert execute_rxinfer_script(script, timeout=7) is True
    assert captured[0]["timeout"] == 7


def test_activeinference_script_timeout_reaches_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-script timeout lands on the subprocess envelope call."""
    captured: list[dict[str, Any]] = []

    def recording_envelope(command: list[Any], **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return _success_envelope()

    monkeypatch.setattr(
        activeinference_runner, "run_subprocess_envelope", recording_envelope
    )
    monkeypatch.setenv("GNN_ALLOW_UNSAFE_EXEC", "1")

    script = tmp_path / "model_activeinference.jl"
    script.write_text("x = 1\n", encoding="utf-8")

    assert (
        execute_activeinference_script(script, setup_environment=False, timeout=55)
        is True
    )
    assert captured[0]["timeout"] == 55


def test_pymdp_passes_timeout_to_safe_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_pymdp_scripts`` forwards its timeout to ``execute_script_safely``."""
    captured: list[dict[str, Any]] = []

    def recording_safe_executor(script_path: Path, **kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return _success_envelope()

    monkeypatch.setattr(
        executor_module, "execute_script_safely", recording_safe_executor
    )

    pymdp_dir = tmp_path / "pymdp"
    pymdp_dir.mkdir()
    (pymdp_dir / "model.py").write_text("print('hello')\n", encoding="utf-8")

    assert (
        run_pymdp_scripts(
            tmp_path,
            tmp_path / "out",
            recursive_search=False,
            verbose=False,
            timeout=13,
        )
        is True
    )
    assert captured[0]["timeout"] == 13


def test_pymdp_uses_shared_default() -> None:
    """PyMDP runners default to the shared runner timeout constant."""
    for function in (
        pymdp_runner.run_pymdp_scripts,
        pymdp_runner.execute_pymdp_script_with_outputs,
    ):
        parameter = inspect.signature(function).parameters["timeout"]
        assert parameter.default == DEFAULT_RUNNER_TIMEOUT_SECONDS
