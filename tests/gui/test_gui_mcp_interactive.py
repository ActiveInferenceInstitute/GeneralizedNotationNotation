#!/usr/bin/env python3
"""Regression tests for the ``process_gui_mcp`` interactive-derivation contract (F1).

``process_gui`` (``gnn.gui.processor``) owns headless derivation: it derives
``headless = not interactive`` from ``kwargs.get("interactive", False)`` and
ignores any explicit ``headless`` kwarg. The MCP wrapper must therefore forward
``interactive = not headless`` (never a literal ``headless`` key); otherwise an
MCP caller passing ``headless=False`` silently stays headless and never gets an
interactive server.

Test 1 fails on the pre-fix wrapper (it forwarded ``headless`` and no
``interactive``); tests 2-3 pin the value mapping and the ``gui_types``
passthrough. No gradio dependency: ``gnn.gui.mcp.process_gui`` is monkeypatched
with a capturing callable, so the step function is never invoked.
"""

from __future__ import annotations

from typing import Any

import pytest

from gnn.gui.mcp import process_gui_mcp


def test_process_gui_mcp_headless_false_forwards_interactive_and_drops_headless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """headless=False must reach process_gui as interactive=True, no headless key.

    Fails pre-fix: the wrapper forwarded {"headless": False} with no
    "interactive" key, so process_gui (which ignores explicit headless) derived
    interactive=False and MCP callers could never get interactive servers.
    """
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr("gnn.gui.mcp.process_gui", capture)

    result = process_gui_mcp(
        target_directory="in",
        output_directory="out",
        headless=False,
    )

    assert result["success"] is True
    assert captured["interactive"] is True
    assert "headless" not in captured


def test_process_gui_mcp_headless_true_forwards_interactive_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """headless=True must reach process_gui as interactive=False."""
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr("gnn.gui.mcp.process_gui", capture)

    result = process_gui_mcp(
        target_directory="in",
        output_directory="out",
        headless=True,
    )

    assert result["success"] is True
    assert captured["interactive"] is False
    assert "headless" not in captured


def test_process_gui_mcp_forwards_gui_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gui_types is passed through to the step function unchanged."""
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr("gnn.gui.mcp.process_gui", capture)

    result = process_gui_mcp(
        target_directory="in",
        output_directory="out",
        gui_types="gui_1,gui_2",
    )

    assert result["success"] is True
    assert captured["gui_types"] == "gui_1,gui_2"
