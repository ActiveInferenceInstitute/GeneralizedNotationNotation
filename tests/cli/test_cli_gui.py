"""Behavior pins for the ``gnn gui`` subcommand.

Covers the F12 integration-surface change that added ``gui`` as the 17th
subcommand:

- ``build_parser`` exposes ``gui`` with ``--target-dir``, ``--output-dir``,
  ``--gui-types``, ``--interactive``, ``--open-browser``, and
  ``--launch-editor`` flags
- dispatch routes ``gnn gui`` to ``_cmd_gui``, which lazily imports
  ``gnn.gui.process_gui``, forwards the parsed options, and maps the
  returned boolean onto the standard exit codes
- a bare ``gnn`` invocation (no subcommand) keeps printing help without
  dispatching to any handler

These tests fail pre-fix: before the change there is no ``gui`` subcommand
to parse and no ``_cmd_gui`` handler to dispatch, so the parse test exits
via argparse and the dispatch tests cannot route.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gnn.cli as cli


def test_gui_parser_flags() -> None:
    """The gui subparser accepts and stores every documented flag."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "gui",
            "--target-dir",
            "t",
            "--output-dir",
            "o",
            "--gui-types",
            "gui_1",
            "--interactive",
            "--launch-editor",
        ]
    )
    assert isinstance(args, argparse.Namespace)
    assert args.target_dir == "t"
    assert args.output_dir == "o"
    assert args.gui_types == "gui_1"
    assert args.interactive is True
    assert args.launch_editor is True
    assert args.open_browser is False


def test_gui_defaults_match_headless_artifact_mode() -> None:
    """Defaults run headless artifacts on input/gnn_files -> output."""
    parser = cli.build_parser()
    args = parser.parse_args(["gui"])
    assert args.target_dir == "input/gnn_files"
    assert args.output_dir == "output"
    assert args.gui_types == "gui_1,gui_2"
    assert args.interactive is False
    assert args.open_browser is False
    assert args.launch_editor is False


@pytest.mark.parametrize("success", [True, False])
def test_gui_dispatch_forwards_kwargs_and_maps_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
) -> None:
    """``gnn gui`` routes to the (patched) process_gui and maps bool→exit."""
    captured: list[dict[str, object]] = []

    def capture(**kwargs: object) -> bool:
        captured.append(kwargs)
        return success

    monkeypatch.setattr("gnn.gui.process_gui", capture)

    rc = cli.main(["gui", "--target-dir", str(tmp_path)])

    assert rc == (cli.EXIT_SUCCESS if success else cli.EXIT_ERROR)
    assert len(captured) == 1
    call = captured[0]
    assert call["target_dir"] == Path(tmp_path)
    assert call["output_dir"] == Path("output")
    assert call["gui_types"] == "gui_1,gui_2"
    assert call["interactive"] is False
    assert call["launch_editor"] is False
    assert call["open_browser"] is False


def test_gui_dispatch_passes_interactive_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive/launch-editor flags reach process_gui untouched."""
    captured: list[dict[str, object]] = []

    def capture(**kwargs: object) -> bool:
        captured.append(kwargs)
        return True

    monkeypatch.setattr("gnn.gui.process_gui", capture)

    rc = cli.main(
        [
            "gui",
            "--target-dir",
            str(tmp_path),
            "--interactive",
            "--open-browser",
            "--launch-editor",
        ]
    )

    assert rc == cli.EXIT_SUCCESS
    assert captured[0]["interactive"] is True
    assert captured[0]["open_browser"] is True
    assert captured[0]["launch_editor"] is True


def test_bare_invocation_prints_help_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No subcommand: help is printed, no handler (gui included) runs."""
    called: list[dict[str, object]] = []

    def capture(**kwargs: object) -> bool:
        called.append(kwargs)
        return True

    monkeypatch.setattr("gnn.gui.process_gui", capture)

    rc = cli.main([])

    assert rc == cli.EXIT_WARNING
    assert called == []
    assert capsys.readouterr().out.strip() != ""
