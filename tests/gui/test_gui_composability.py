#!/usr/bin/env python3
"""
GUI Composability Tests

Pins the shared, typed building blocks of the GUI module:
- normalize_gui_types / summarize_gui_results (pure functions)
- backend atomic writers (write_text_atomically / write_json_atomically)
- runner plumbing (resolve_output_root, load_first_markdown, launch helper)
- collect_pipeline_outputs + HTML escaping in generate_html_navigation
- process_gui honoring a caller-provided logger
- gui_3 using the shared gradio backend detection (reload contract)
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from gnn.gui import (
    MAX_FILES_PER_SECTION,
    collect_pipeline_outputs,
    generate_html_navigation,
    normalize_gui_types,
    process_gui,
    summarize_gui_results,
)


def get_real_logger() -> Any:
    """Create a real logger that captures output to a StringIO stream."""
    logger = logging.getLogger("test_gui_composability_logger")
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)

    def get_messages(level: Any = None) -> Any:
        content = stream.getvalue().splitlines()
        if not level:
            return [line.split(":", 1)[1] for line in content if ":" in line]
        level_str = level.upper()
        return [
            line.split(":", 1)[1]
            for line in content
            if line.startswith(f"{level_str}:")
        ]

    logger.get_messages = get_messages  # type: ignore[attr-defined]
    return logger


class TestNormalizeGuiTypes:
    """normalize_gui_types parses strings, lists, and defaults."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_none_yields_pipeline_default(self) -> None:
        assert normalize_gui_types(None) == ["gui_1", "gui_2"]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_comma_string_is_split_and_stripped(self) -> None:
        assert normalize_gui_types(" gui_3 , oxdraw ,") == ["gui_3", "oxdraw"]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_blank_entries_are_dropped(self) -> None:
        assert normalize_gui_types("gui_1,,  ,gui_2") == ["gui_1", "gui_2"]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_list_passthrough_is_stripped(self) -> None:
        assert normalize_gui_types(["gui_1", " oxdraw "]) == ["gui_1", "oxdraw"]


class TestSummarizeGuiResults:
    """summarize_gui_results aggregates per-GUI outcomes into a typed summary."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_mixed_results_are_aggregated(self) -> None:
        summary = summarize_gui_results(
            {
                "gui_1": {"success": True},
                "gui_2": {"success": False, "error": "boom"},
                "oxdraw": {"success": True},
            }
        )
        assert summary["total"] == 3
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1
        assert summary["failed_guis"] == ["gui_2"]
        assert summary["overall_success"] is False

    @pytest.mark.unit
    @pytest.mark.fast
    def test_missing_success_key_counts_as_failure(self) -> None:
        summary = summarize_gui_results({"gui_3": {"note": "no success field"}})
        assert summary["failed_guis"] == ["gui_3"]
        assert summary["overall_success"] is False

    @pytest.mark.unit
    @pytest.mark.fast
    def test_empty_results_are_all_success(self) -> None:
        summary = summarize_gui_results({})
        assert summary == {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "failed_guis": [],
            "overall_success": True,
        }

    @pytest.mark.unit
    @pytest.mark.fast
    def test_matches_saved_pipeline_summary(self, isolated_temp_dir: Any) -> None:
        """Integration: aggregation agrees with a real headless process_gui run."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        assert process_gui(target_dir=target, output_dir=output, headless=True)

        saved = json.loads((output / "gui_processing_summary.json").read_text())
        summary = summarize_gui_results(saved["results"])
        assert summary["overall_success"] is saved["overall_success"]
        assert summary["total"] == len(saved["results"])


class TestAtomicWriters:
    """backend atomic writers create parents and replace content."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_write_text_atomically_creates_and_overwrites(
        self, isolated_temp_dir: Any
    ) -> None:
        from gnn.gui.backend import write_text_atomically

        path = isolated_temp_dir / "nested" / "dir" / "artifact.md"
        write_text_atomically(path, "first")
        assert path.read_text() == "first"
        write_text_atomically(path, "second")
        assert path.read_text() == "second"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_write_json_atomically_writes_indented_json(
        self, isolated_temp_dir: Any
    ) -> None:
        from gnn.gui.backend import write_json_atomically

        path = isolated_temp_dir / "payload.json"
        write_json_atomically(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
        assert "\n" in path.read_text()  # indent=2 formatting


class TestRunnerPlumbing:
    """Shared runner helpers behave deterministically."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_resolve_output_root_falls_back_without_pipeline(
        self, isolated_temp_dir: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.gui.runner import resolve_output_root

        monkeypatch.setitem(sys.modules, "gnn.pipeline.config", None)
        out = isolated_temp_dir / "out"
        assert resolve_output_root(out) == out

    @pytest.mark.unit
    @pytest.mark.fast
    def test_resolve_output_root_uses_pipeline_mapping(
        self, isolated_temp_dir: Any
    ) -> None:
        from gnn.gui.runner import resolve_output_root

        pipeline_config = importlib.import_module("gnn.pipeline.config")
        get_output_dir_for_script = pipeline_config.get_output_dir_for_script

        out = isolated_temp_dir / "out"
        expected = Path(get_output_dir_for_script("22_gui.py", out))
        assert resolve_output_root(out) == expected

    @pytest.mark.unit
    @pytest.mark.fast
    def test_load_first_markdown_prefers_patterns_then_top_level(
        self, isolated_temp_dir: Any
    ) -> None:
        from gnn.gui.runner import load_first_markdown

        target = isolated_temp_dir / "input"
        target.mkdir()
        (target / "aaa.md").write_text("plain")
        (target / "zzz_pomdp.md").write_text("pomdp")
        (target / "sub").mkdir()
        (target / "sub" / "nested.md").write_text("nested")

        assert load_first_markdown(target, prefer_patterns=("*pomdp*.md",)) == "pomdp"
        assert load_first_markdown(target) == "plain"

        (target / "aaa.md").unlink()
        (target / "zzz_pomdp.md").unlink()
        assert load_first_markdown(target) == "nested"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_load_first_markdown_empty_dir_returns_none(
        self, isolated_temp_dir: Any
    ) -> None:
        from gnn.gui.runner import load_first_markdown

        assert load_first_markdown(isolated_temp_dir / "missing") is None

    @pytest.mark.unit
    @pytest.mark.fast
    def test_launch_gradio_in_thread_forwards_launch_kwargs(self) -> None:
        from gnn.gui.runner import launch_gradio_in_thread

        calls: list[dict[str, Any]] = []

        class FakeDemo:
            def launch(self, **kwargs: Any) -> None:
                calls.append(kwargs)

        thread = launch_gradio_in_thread(FakeDemo(), port=7865, open_browser=False)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(calls) == 1
        assert calls[0]["server_port"] == 7865
        assert calls[0]["server_name"] == "0.0.0.0"  # nosec B104
        assert calls[0]["inbrowser"] is False


class TestCollectPipelineOutputs:
    """collect_pipeline_outputs discovers and caps artifacts per step."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_discovers_files_and_skips_missing_dirs(
        self, isolated_temp_dir: Any
    ) -> None:
        pipeline_output = isolated_temp_dir / "output"
        gnn_dir = pipeline_output / "3_gnn_output"
        gnn_dir.mkdir(parents=True)
        (gnn_dir / "b.json").write_text("{}")
        (gnn_dir / "a.json").write_text("{}")
        (gnn_dir / "ignored.txt").write_text("not matched")

        sections, total = collect_pipeline_outputs(pipeline_output)
        assert total == 2
        assert len(sections) == 1
        section = sections[0]
        assert section["step_dir"] == "3_gnn_output"
        assert section["file_count"] == 2
        assert [f["name"] for f in section["files"]] == ["a.json", "b.json"]

    @pytest.mark.unit
    @pytest.mark.fast
    def test_caps_listing_but_counts_all_files(self, isolated_temp_dir: Any) -> None:
        pipeline_output = isolated_temp_dir / "output"
        gnn_dir = pipeline_output / "3_gnn_output"
        gnn_dir.mkdir(parents=True)
        for index in range(MAX_FILES_PER_SECTION + 5):
            (gnn_dir / f"f{index:02d}.json").write_text("{}")

        sections, total = collect_pipeline_outputs(pipeline_output)
        assert total == MAX_FILES_PER_SECTION + 5
        assert sections[0]["file_count"] == MAX_FILES_PER_SECTION + 5
        assert len(sections[0]["files"]) == MAX_FILES_PER_SECTION


class TestNavigationEscaping:
    """navigation.html escapes file names and paths."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_file_names_are_html_escaped(self, isolated_temp_dir: Any) -> None:
        pipeline_output = isolated_temp_dir / "output"
        gui_output = pipeline_output / "22_gui_output"
        gui_output.mkdir(parents=True)
        hostile = pipeline_output / "3_gnn_output"
        hostile.mkdir()
        (hostile / 'a"<b>&c.json').write_text("{}")

        assert generate_html_navigation(
            pipeline_output, gui_output, logging.getLogger("test_nav")
        )

        content = (gui_output / "navigation.html").read_text()
        assert "a&quot;&lt;b&gt;&amp;c.json" in content
        # the hostile name must never appear raw inside an attribute or element
        assert 'a"<b>&c.json' not in content


class TestProcessGuiLogger:
    """process_gui honors a caller-provided logger (documented API)."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_caller_logger_receives_messages(self, isolated_temp_dir: Any) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        result = process_gui(
            target_dir=target, output_dir=output, logger=logger, headless=True
        )

        assert result is True
        messages = logger.get_messages()
        assert any("Running GUI types" in message for message in messages)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_non_logger_kwarg_is_ignored_safely(self, isolated_temp_dir: Any) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        assert (
            process_gui(
                target_dir=target,
                output_dir=output,
                logger="not-a-logger",
                headless=True,
            )
            is True
        )


class TestGUI3SharedBackendDetection:
    """gui_3 uses the shared backend detector with the same reload contract."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_dummy_gradio_module_yields_none_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gui3_processor = importlib.import_module("gnn.gui.gui_3.processor")

        original_gradio = sys.modules.get("gradio")
        monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
        try:
            reloaded = importlib.reload(gui3_processor)
            assert reloaded._GUI_BACKEND is None
        finally:
            if original_gradio is None:
                monkeypatch.delitem(sys.modules, "gradio", raising=False)
            else:
                monkeypatch.setitem(sys.modules, "gradio", original_gradio)
            importlib.reload(gui3_processor)
