"""Pins for ``utils/visual_logging.VisualLogger`` (previously untested)."""

from __future__ import annotations

from gnn.utils.observability.visual_logging import VisualConfig, VisualLogger


def test_format_message_strips_status_icons_when_disabled() -> None:
    logger = VisualLogger("w2test", VisualConfig(enable_emoji=False))

    formatted = logger.format_message("done ✅ something ℹ️")

    assert "✅" not in formatted
    assert "ℹ️" not in formatted
    assert formatted == "done  something "


def test_format_message_prepends_correlation_and_timestamp() -> None:
    config = VisualConfig(show_correlation_ids=True, show_timestamps=True)
    logger = VisualLogger("w2test", config)
    logger.set_correlation_id("corr-42")

    formatted = logger.format_message("step body")

    assert "[corr-42]" in formatted
    assert formatted.endswith("step body")


def test_format_message_plain_when_all_decorations_off() -> None:
    config = VisualConfig(show_correlation_ids=False, show_timestamps=False)
    logger = VisualLogger("w2test", config)

    assert logger.format_message("plain text") == "plain text"


def test_print_summary_writes_rows_to_stdout(capsys: object) -> None:
    logger = VisualLogger("w2test")

    logger.print_summary("Wave Summary", {"files": 12, "violations": 0})

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Wave Summary" in captured.out
    assert "files" in captured.out
    assert "12" in captured.out


def test_print_header_writes_border_block(capsys: object) -> None:
    logger = VisualLogger("w2test")

    logger.print_header("SECTION TITLE", width=40)

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "SECTION TITLE" in captured.out
    # Rich box-drawing border (or ASCII '=' fallback when rich is absent).
    assert "╭" in captured.out or "=" in captured.out
