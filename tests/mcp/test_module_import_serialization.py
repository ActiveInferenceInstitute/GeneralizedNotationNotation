"""Real-behavior tests for MCP module-import serialization.

The MCP discovery executor loads ``gnn.<module>.mcp`` modules on worker
threads. Several module bodies call ``matplotlib.use(...)`` at import time;
a concurrent ``matplotlib.use`` while another thread is still executing
``matplotlib.pyplot``'s module body raises "partially initialized module
'matplotlib.pyplot' has no attribute 'switch_backend'" — observed as a
flaky CI failure on fresh runners (2026-09-07, release tip and PR CI).
The loader imports every module under ``MCP._module_import_lock``; these
tests pin that invariant.
"""

from __future__ import annotations

import importlib
import time
import types
from pathlib import Path
from typing import Any

import pytest

from gnn.mcp import MCP

pytestmark = pytest.mark.mcp


def _write_module(directory: Path, name: str) -> Path:
    """Create a minimal discoverable module with an mcp.py entry point."""
    pkg = directory / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    mcp_file = pkg / "mcp.py"
    mcp_file.write_text(
        '"""Stub MCP module."""\n'
        "def register_tools(server):\n"
        "    return None\n",
        encoding="utf-8",
    )
    return mcp_file


def _instrumented_import(real_import: Any, log: dict[str, int]) -> Any:
    """Wrap importlib.import_module to record concurrent in-flight imports."""

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("gnn.serialized_"):
            log["in_flight"] += 1
            log["max_in_flight"] = max(log["max_in_flight"], log["in_flight"])
            try:
                time.sleep(0.05)
                stub = types.ModuleType(name)
                stub.register_tools = lambda _server: None  # type: ignore[attr-defined]
                return stub
            finally:
                log["in_flight"] -= 1
        return real_import(name, *args, **kwargs)

    return fake_import


def test_module_imports_never_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent discovery must not execute two module imports at once.

    Serializing module imports is what makes ``matplotlib.use`` inside a
    module body safe against a sibling thread's mid-init pyplot import.
    """
    mcp_files = {
        name: _write_module(tmp_path, name)
        for name in ("serialized_a", "serialized_b", "serialized_c")
    }

    log = {"in_flight": 0, "max_in_flight": 0}
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", _instrumented_import(real_import, log)
    )

    server = MCP()
    assert server._executor is not None

    futures = [
        server._executor.submit(server._load_module, tmp_path / name, mcp_file)
        for name, mcp_file in mcp_files.items()
    ]
    for future in futures:
        assert future.result(timeout=30) is True

    assert log["max_in_flight"] == 1, (
        "two module imports overlapped; the matplotlib.use vs pyplot-init "
        "race is no longer excluded"
    )


def test_serialized_import_still_registers_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serialization must not break the load path: a stub module loads True."""
    mcp_file = _write_module(tmp_path, "serialized_single")

    log = {"in_flight": 0, "max_in_flight": 0}
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", _instrumented_import(real_import, log)
    )

    server = MCP()
    loaded = server._load_module(tmp_path / "serialized_single", mcp_file)

    assert loaded is True
    assert server.modules["serialized_single"].status == "loaded"
