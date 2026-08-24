#!/usr/bin/env python3
"""
Regression tests for gnn.watcher — GNNWatcher CLI watch mode.

Pins the start/stop + polling discovery lifecycle, debounced firing, the
default validation callback (valid vs. invalid GNN), extension filtering, and
the robustness guarantee that a misbehaving callback cannot terminate the
watch loop.
"""

import sys
import threading
import time
from pathlib import Path
from typing import Any, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.watcher import GNNWatcher  # noqa: E402

_VALID_GNN = (
    "## GNNSection\nGNN-2025-10\n"
    "## GNNVersionAndFlags\nversion=1\n"
    "## ModelName\ntest_model\n"
    "## StateSpaceBlock\ns[1,1]\no[1,1]\n"
    "## Connections\ns->o\n"
)

_INVALID_GNN = (
    "## GNNVersionAndFlags\nversion=1\n"
    "## ModelName\ntest_model\n"
    "## Connections\ns->o\n"
)


class TestDefaultCallback:
    """Coverage of the default validation callback (real gnn.schema path)."""

    @pytest.mark.unit
    def test_valid_file_prints_valid(self, tmp_path: Path, capsys: Any) -> None:
        path = tmp_path / "ok.gnn"
        path.write_text(_VALID_GNN)
        GNNWatcher._default_callback(path, _VALID_GNN)
        captured = capsys.readouterr().out
        assert "valid" in captured

    @pytest.mark.unit
    def test_invalid_file_reports_issues(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        path = tmp_path / "bad.gnn"
        path.write_text(_INVALID_GNN)
        GNNWatcher._default_callback(path, _INVALID_GNN)
        captured = capsys.readouterr().out
        assert "issue" in captured
        assert "GNN-E001" in captured


class TestDebouncedFire:
    """Coverage of the debouncing mechanism on a single file."""

    @pytest.mark.unit
    def test_fires_once_within_debounce_window(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("# A")
        seen: List[Path] = []
        w = GNNWatcher(tmp_path, on_change=lambda p, c: seen.append(p), debounce_ms=50000)
        w._debounced_fire(f)
        w._debounced_fire(f)  # suppressed inside the debounce window
        assert len(seen) == 1

    @pytest.mark.unit
    def test_fires_again_after_window(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("# A")
        seen: List[Path] = []
        w = GNNWatcher(tmp_path, on_change=lambda p, c: seen.append(p), debounce_ms=0)
        w._debounced_fire(f)
        time.sleep(0.01)
        w._debounced_fire(f)
        assert len(seen) == 2

    @pytest.mark.unit
    def test_callback_receives_file_content(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("payload-here")
        got: List[Tuple[Path, str]] = []
        w = GNNWatcher(tmp_path, on_change=lambda p, c: got.append((p, c)))
        w._debounced_fire(f)
        assert got and got[0][1] == "payload-here"
        assert got[0][0] == f

    @pytest.mark.unit
    def test_misbehaving_callback_does_not_raise(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("# A")

        def boom(path: Path, content: str) -> None:
            raise RuntimeError("callback exploded")

        w = GNNWatcher(tmp_path, on_change=boom)
        # Must swallow the callback failure rather than propagate it.
        w._debounced_fire(f)


class TestPollingLifecycle:
    """End-to-end polling watcher lifecycle: discovery, filtering, stop."""

    @pytest.mark.unit
    def test_discovers_new_file_and_filters_extensions(self, tmp_path: Path) -> None:
        interested = threading.Event()
        observed: List[Path] = []

        def on_change(path: Path, content: str) -> None:
            observed.append(path)
            interested.set()

        w = GNNWatcher(
            tmp_path, on_change=on_change, debounce_ms=0, extensions=(".md",)
        )
        thread = threading.Thread(
            target=w._start_polling, kwargs={"interval": 0.05}, daemon=True
        )
        thread.start()
        try:
            time.sleep(0.15)
            (tmp_path / "model.md").write_text(_VALID_GNN)
            assert interested.wait(4.0), "watcher never fired for new .md file"
            # Non-watched extension must not trigger a callback.
            (tmp_path / "notes.txt").write_text("ignored")
            time.sleep(0.2)
            assert not any(p.suffix == ".txt" for p in observed)
            assert any(p.name == "model.md" for p in observed)
        finally:
            w.stop()
            thread.join(2.0)
            assert not thread.is_alive()

    @pytest.mark.unit
    def test_start_then_stop_returns_cleanly(self, tmp_path: Path) -> None:
        w = GNNWatcher(tmp_path, on_change=lambda p, c: None)
        thread = threading.Thread(
            target=w._start_polling, kwargs={"interval": 0.05}, daemon=True
        )
        thread.start()
        time.sleep(0.1)
        w.stop()
        thread.join(2.0)
        assert not thread.is_alive()