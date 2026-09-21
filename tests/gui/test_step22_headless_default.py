#!/usr/bin/env python3
"""Step 22 headless-default and daemon-runner regression tests (FE#4).

Covers the derived ``headless = not interactive`` contract of
``process_gui`` (neither flag given, or the pipeline/CLI defaults, must run
HEADLESS; interactive still wins) and the daemon server-thread contract of
``gnn.gui.runner.launch_gradio_in_thread`` (daemons never block exit; the
``process_gui`` keep-alive is gated strictly on server liveness).

No gradio dependency: gradio is stubbed to a bare module and the gui_1/gui_2
processors are reloaded so their backend detection falls back to the static
headless path, exactly like
``tests/gui/test_gui_functionality.py::test_importable_gradio_without_blocks_falls_back_without_failing``.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gnn.gui import process_gui
from gnn.gui import runner as gui_runner


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("test_step22_headless_default")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def stubbed_gradio() -> Iterator[None]:
    """Stub gradio to a bare module so gui_1/gui_2 fall back to static mode."""
    gui1_processor = importlib.import_module("gnn.gui.gui_1.processor")
    gui2_processor = importlib.import_module("gnn.gui.gui_2.processor")

    original_gradio = sys.modules.get("gradio")
    sys.modules["gradio"] = types.ModuleType("gradio")
    importlib.reload(gui1_processor)
    importlib.reload(gui2_processor)
    try:
        yield
    finally:
        if original_gradio is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = original_gradio
        importlib.reload(gui1_processor)
        importlib.reload(gui2_processor)


@pytest.fixture
def clean_server_registry() -> Iterator[None]:
    """Isolate the runner's global server-thread registry per test."""
    gui_runner.clear_launched_server_threads()
    try:
        yield
    finally:
        gui_runner.clear_launched_server_threads()


def _write_target(isolated_temp_dir: Path) -> tuple[Path, Path]:
    target = isolated_temp_dir / "input"
    output = isolated_temp_dir / "output"
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.md").write_text("# Test Model\n")
    return target, output


def _load_summary(output: Path) -> dict[str, Any]:
    summary_file = output / "gui_processing_summary.json"
    assert summary_file.exists()
    summary: dict[str, Any] = json.loads(summary_file.read_text())
    return summary


class TestStep22HeadlessDefault:
    """``process_gui`` must derive headless = not interactive."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_default_call_with_no_mode_kwargs_is_headless(
        self, isolated_temp_dir: Path, stubbed_gradio: None, clean_server_registry: None
    ) -> None:
        target, output = _write_target(isolated_temp_dir)

        result = process_gui(
            target_dir=target, output_dir=output, logger=_test_logger()
        )

        assert result is True
        summary = _load_summary(output)
        assert summary["overall_success"] is True
        assert summary["mode"] == "headless"
        assert gui_runner.registered_server_threads() == ()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_pipeline_default_kwargs_run_headless(
        self, isolated_temp_dir: Path, stubbed_gradio: None, clean_server_registry: None
    ) -> None:
        """The step-config defaults (headless=False, interactive=False) are HEADLESS.

        Pre-fix these defaults forced the interactive path: with real gradio
        gui_1/gui_2 launched non-daemon servers and the process hung; with
        gradio stubbed away the summary still reported mode "interactive".
        """
        target, output = _write_target(isolated_temp_dir)

        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            headless=False,
            interactive=False,
            gui_types="gui_1,gui_2",
        )

        assert result is True
        summary = _load_summary(output)
        assert summary["overall_success"] is True
        assert summary["mode"] == "headless"
        assert gui_runner.registered_server_threads() == ()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_interactive_true_keeps_interactive_mode(
        self, isolated_temp_dir: Path, stubbed_gradio: None, clean_server_registry: None
    ) -> None:
        target, output = _write_target(isolated_temp_dir)

        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            interactive=True,
            gui_types="gui_1,gui_2",
        )

        assert result is True
        summary = _load_summary(output)
        assert summary["mode"] == "interactive"
        # Gradio-less backends never launch servers, so nothing registered.
        assert gui_runner.registered_server_threads() == ()


class TestRunnerDaemonLaunch:
    """``launch_gradio_in_thread`` returns daemon threads and registers them."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_launched_thread_is_daemon_and_registered(
        self, clean_server_registry: None
    ) -> None:
        launched = threading.Event()

        class FakeDemo:
            def launch(self, **kwargs: Any) -> None:
                launched.set()

        thread = gui_runner.launch_gradio_in_thread(
            FakeDemo(), port=7860, open_browser=False
        )

        assert thread.daemon is True
        assert launched.wait(timeout=5)
        assert thread in gui_runner.registered_server_threads()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_interactive_servers_running_tracks_liveness(
        self, clean_server_registry: None
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingDemo:
            def launch(self, **kwargs: Any) -> None:
                started.set()
                release.wait(timeout=10)

        gui_runner.launch_gradio_in_thread(
            BlockingDemo(), port=7861, open_browser=False
        )

        assert started.wait(timeout=5)
        assert gui_runner.interactive_servers_running() is True

        release.set()
        deadline = time.monotonic() + 10
        while gui_runner.interactive_servers_running() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert gui_runner.interactive_servers_running() is False

    @pytest.mark.unit
    @pytest.mark.fast
    def test_interactive_servers_running_false_with_no_servers(
        self, clean_server_registry: None
    ) -> None:
        assert gui_runner.interactive_servers_running() is False


class TestInteractiveKeepAlive:
    """``process_gui`` keep-alive is gated strictly on server liveness."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_interactive_without_live_servers_returns_without_blocking(
        self, isolated_temp_dir: Path, stubbed_gradio: None, clean_server_registry: None
    ) -> None:
        target, output = _write_target(isolated_temp_dir)

        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            interactive=True,
            gui_types="gui_1,gui_2",
        )

        assert result is True
        assert _load_summary(output)["overall_success"] is True

    @pytest.mark.unit
    @pytest.mark.fast
    def test_interactive_blocks_while_server_live_then_returns(
        self, isolated_temp_dir: Path, stubbed_gradio: None, clean_server_registry: None
    ) -> None:
        """process_gui stays alive while a registered server serves."""
        target, output = _write_target(isolated_temp_dir)
        started = threading.Event()
        release = threading.Event()

        class BlockingDemo:
            def launch(self, **kwargs: Any) -> None:
                started.set()
                release.wait(timeout=20)

        gui_runner.launch_gradio_in_thread(
            BlockingDemo(), port=7862, open_browser=False
        )
        assert started.wait(timeout=5)

        result_holder: dict[str, Any] = {}

        def run() -> None:
            result_holder["result"] = process_gui(
                target_dir=target,
                output_dir=output,
                logger=_test_logger(),
                interactive=True,
                gui_types="gui_1,gui_2",
            )

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        # While the sentinel server is live, the keep-alive must hold the
        # worker open (pre-fix process_gui returned immediately here).
        worker.join(timeout=1.2)
        assert worker.is_alive()

        release.set()
        worker.join(timeout=20)
        assert not worker.is_alive()
        assert result_holder["result"] is True
