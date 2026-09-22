"""Tests for the executor's lazy runner-import registry.

Importing ``gnn.execute`` must not bloom heavy optional backends (jax, torch,
discopy, matplotlib, networkx, numpyro); runner modules load only when the
framework registry is consulted, and ``_runner_state`` is the single seam
tests patch to force a backend unavailable.
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

import gnn.execute.executor as executor_module
from gnn.execute.executor import GNNExecutor, _RunnerState, list_frameworks

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Backend modules that must stay out of ``sys.modules`` until a framework
# registry lookup actually needs them.
_HEAVY_BACKEND_MODULES = (
    "jax",
    "discopy",
    "matplotlib",
    "networkx",
    "torch",
    "numpyro",
)

_COLD_IMPORT_SNIPPET = (
    "import sys; "
    "import gnn.execute; "
    "import gnn.execute.executor; "
    f"mods = {list(_HEAVY_BACKEND_MODULES)!r}; "
    'print(",".join(m for m in mods if m in sys.modules))'
)


def test_cold_import_blooms_no_heavy_backends() -> None:
    """Importing the executor in a fresh interpreter imports no heavy backend."""
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _COLD_IMPORT_SNIPPET],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.stdout.strip() == ""


def test_runner_state_is_cached() -> None:
    """Repeated ``_runner_state`` lookups return one cached verdict object."""
    first = executor_module._runner_state("pymdp")
    second = executor_module._runner_state("pymdp")
    assert isinstance(first, _RunnerState)
    assert first is second


def test_list_frameworks_shape() -> None:
    """``list_frameworks`` reports every registered backend in canonical order."""
    records = list_frameworks()
    assert len(records) == len(executor_module.FRAMEWORK_DIR_NAMES)
    for record in records:
        assert set(record) == {"framework", "result_key", "available", "operation"}
        assert isinstance(record["available"], bool)
    assert [record["framework"] for record in records] == list(
        executor_module.FRAMEWORK_DIR_NAMES
    )


def test_list_frameworks_loads_runners_lazily() -> None:
    """Registry use — not module import — is what pulls runner modules in."""
    executor_module._runner_state.cache_clear()
    executor_module.list_frameworks()
    assert "gnn.execute.discopy.discopy_executor" in sys.modules


def test_unavailable_runner_state_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """A patched unavailable ``_runner_state`` makes lean dispatch fail closed."""
    monkeypatch.setattr(
        executor_module,
        "_runner_state",
        lambda key: _RunnerState(available=False, runner=None),
    )
    result = GNNExecutor()._execute_lean_verification("model.md")
    assert result == {"success": False, "error": "fep_lean unavailable"}
