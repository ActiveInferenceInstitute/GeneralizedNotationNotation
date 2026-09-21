"""Behavior tests for the documented API server entrypoints.

The documented boot commands are ``python -m gnn.api.server`` and
``uvicorn gnn.api.server:app`` (see ``src/gnn/api/server.py`` and
``src/gnn/api/README.md``). A regression that breaks either path — e.g. a
stale flat ``api.server`` import string handed to ``uvicorn.run`` — must
fail these tests.
"""

import importlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC.parent


def _free_port() -> int:
    """Reserve an ephemeral loopback port for the subprocess boot."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_run_server_hands_uvicorn_an_importable_app() -> Any:
    """run_server() must pass uvicorn an import string that resolves to a
    module exposing the ASGI ``app``; this is how ``python -m gnn.api.server``
    boots its server process."""
    from gnn.api import server

    captured: dict[str, Any] = {}

    def fake_run(app_target: str, **kwargs: Any) -> None:
        captured["app_target"] = app_target
        captured["kwargs"] = kwargs

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(server.uvicorn, "run", fake_run)
        server.run_server(host="127.0.0.1", port=_free_port())
    finally:
        monkeypatch.undo()

    target = captured["app_target"]
    module_name, _, attr = target.partition(":")
    assert attr, f"uvicorn.run target must be 'module:app', got {target!r}"
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), (
        f"uvicorn.run target {target!r} does not resolve to an ASGI app"
    )
    assert captured["kwargs"]["host"] == "127.0.0.1"


def test_python_m_entrypoint_boots_and_serves_health() -> None:
    """``python -m gnn.api.server`` must boot and answer /api/v1/health.

    Regression context: a stale flat import string in run_server() crashed
    every documented boot with ModuleNotFoundError right after argparse.
    """
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.Popen(
        [sys.executable, "-m", "gnn.api.server", "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        healthy = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/v1/health", timeout=2
                ) as response:
                    if response.status == 200:
                        healthy = True
                        break
            except OSError:
                time.sleep(0.2)
        if not healthy:
            try:
                output, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                output, _ = proc.communicate()
            pytest.fail(
                "python -m gnn.api.server never served /api/v1/health "
                f"(exit code {proc.returncode}); boot output tail:\n{(output or '')[-2000:]}"
            )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
