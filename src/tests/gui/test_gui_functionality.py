#!/usr/bin/env python3
"""
GUI Functionality Tests

Tests the GUI module's process_gui function and related functionality:
- Headless mode execution
- Interactive mode configuration
- Output artifact generation
- HTML navigation generation
- Error handling
"""

import importlib
import io
import json
import logging
import sys
import types
from typing import Any, cast

import pytest

from gui import generate_html_navigation, process_gui


def get_real_logger() -> Any:
    """Create a real logger that captures output to a StringIO stream."""
    logger = logging.getLogger("test_gui_logger")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicate logs in parametrized tests
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(handler)

    logger_state = cast(Any, logger)
    logger_state.stream = stream

    # Helper to retrieve trapped messages exactly like the simulated
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

    logger_state.get_messages = get_messages
    return logger


class TestGUIHeadlessMode:
    """Tests for GUI headless mode execution."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_mode_returns_success(self, isolated_temp_dir: Any) -> Any:
        """Test that headless mode returns success for valid input."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            verbose=True,
            headless=True,
        )
        assert result is True

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_mode_creates_summary(self, isolated_temp_dir: Any) -> Any:
        """Test that headless mode creates processing summary."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(target_dir=target, output_dir=output, logger=logger, headless=True)

        summary_file = output / "gui_processing_summary.json"
        assert summary_file.exists()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_summary_has_correct_mode(self, isolated_temp_dir: Any) -> Any:
        """Test that processing summary reports correct mode."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(target_dir=target, output_dir=output, logger=logger, headless=True)

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())
        assert summary.get("mode") == "headless"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_mode_empty_directory(self, isolated_temp_dir: Any) -> Any:
        """Test headless mode with empty input directory."""
        target = isolated_temp_dir / "empty_input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)

        logger = get_real_logger()
        result = process_gui(
            target_dir=target, output_dir=output, logger=logger, headless=True
        )
        # Should handle empty directory gracefully
        assert isinstance(result, bool)


class TestGUIConfiguration:
    """Tests for GUI configuration options."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui_types_string_parsing(self, isolated_temp_dir: Any) -> Any:
        """Test that GUI types string is parsed correctly."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True,
            gui_types="gui_1",
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())
        assert "gui_1" in summary.get("gui_types", [])

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui_types_list(self, isolated_temp_dir: Any) -> Any:
        """Test that GUI types can be passed as list."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True,
            gui_types=["gui_1", "gui_2"],
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())
        assert "gui_1" in summary.get("gui_types", [])
        assert "gui_2" in summary.get("gui_types", [])

    @pytest.mark.unit
    @pytest.mark.fast
    def test_unknown_gui_type_handled(self, isolated_temp_dir: Any, caplog: Any) -> Any:
        """Test that unknown GUI types are handled gracefully."""
        import logging

        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)  # Create output dir
        (target / "model.md").write_text("# Test Model\n")

        with caplog.at_level(logging.WARNING):
            process_gui(
                target_dir=target,
                output_dir=output,
                headless=True,
                gui_types="nonexistent_gui",
            )

        # Should have warning about unknown GUI type in captured logs
        assert any(
            "Unknown" in r.message or "nonexistent" in r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    @pytest.mark.unit
    @pytest.mark.fast
    def test_importable_gradio_without_blocks_falls_back_without_failing(
        self, isolated_temp_dir: Any, monkeypatch: pytest.MonkeyPatch
    ) -> Any:
        """Importable Gradio without Blocks should use recovery artifacts."""
        gui1_processor = importlib.import_module("gui.gui_1.processor")
        gui2_processor = importlib.import_module("gui.gui_2.processor")

        original_gradio = sys.modules.get("gradio")
        dummy_gradio = types.ModuleType("gradio")
        monkeypatch.setitem(sys.modules, "gradio", dummy_gradio)
        importlib.reload(gui1_processor)
        importlib.reload(gui2_processor)

        try:
            assert gui1_processor._GUI_BACKEND is None
            assert gui2_processor._GUI_BACKEND is None

            target = isolated_temp_dir / "input"
            output = isolated_temp_dir / "output"
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.md").write_text("# Test Model\n")

            logger = get_real_logger()
            result = process_gui(
                target_dir=target,
                output_dir=output,
                logger=logger,
                interactive=True,
                gui_types="gui_1,gui_2",
            )

            assert result is True
            summary = json.loads((output / "gui_processing_summary.json").read_text())
            assert summary["overall_success"] is True
            assert summary["results"]["gui_1"]["success"] is True
            assert summary["results"]["gui_2"]["success"] is True
            assert summary["results"]["gui_1"]["backend"] == "none"
            assert summary["results"]["gui_2"]["backend"] == "none"
            assert summary["results"]["gui_2"]["status"] == "static_headless_mode"
        finally:
            if original_gradio is None:
                sys.modules.pop("gradio", None)
            else:
                sys.modules["gradio"] = original_gradio
            importlib.reload(gui1_processor)
            importlib.reload(gui2_processor)


class TestGUIHTMLNavigation:
    """Tests for HTML navigation generation."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_generate_html_navigation_creates_file(self, isolated_temp_dir: Any) -> Any:
        """Test that HTML navigation file is created."""
        pipeline_output = isolated_temp_dir / "output"
        gui_output = pipeline_output / "22_gui_output"
        gui_output.mkdir(parents=True, exist_ok=True)

        # Create some test output directories
        (pipeline_output / "3_gnn_output").mkdir()
        (pipeline_output / "3_gnn_output" / "test.json").write_text("{}")

        logger = get_real_logger()
        result = generate_html_navigation(pipeline_output, gui_output, logger)

        nav_file = gui_output / "navigation.html"
        assert nav_file.exists()
        assert result is True

    @pytest.mark.unit
    @pytest.mark.fast
    def test_html_navigation_contains_structure(self, isolated_temp_dir: Any) -> Any:
        """Test that HTML navigation has proper structure."""
        pipeline_output = isolated_temp_dir / "output"
        gui_output = pipeline_output / "22_gui_output"
        gui_output.mkdir(parents=True, exist_ok=True)

        (pipeline_output / "3_gnn_output").mkdir()
        (pipeline_output / "3_gnn_output" / "test.json").write_text("{}")

        logger = get_real_logger()
        generate_html_navigation(pipeline_output, gui_output, logger)

        nav_file = gui_output / "navigation.html"
        content = nav_file.read_text()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "GNN Pipeline" in content

    @pytest.mark.unit
    @pytest.mark.fast
    def test_html_navigation_empty_output(self, isolated_temp_dir: Any) -> Any:
        """Test HTML navigation with no output directories."""
        pipeline_output = isolated_temp_dir / "empty_output"
        gui_output = pipeline_output / "22_gui_output"
        gui_output.mkdir(parents=True, exist_ok=True)

        logger = get_real_logger()
        result = generate_html_navigation(pipeline_output, gui_output, logger)

        # Should still create navigation file
        assert result is True


class TestGUIOutputArtifacts:
    """Tests for GUI output artifact generation."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_process_gui_creates_output_directory(self, isolated_temp_dir: Any) -> Any:
        """Test that process_gui creates output directory if needed."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "nonexistent" / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(target_dir=target, output_dir=output, logger=logger, headless=True)

        assert output.exists()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_summary_contains_results(self, isolated_temp_dir: Any) -> Any:
        """Test that summary contains results for each GUI type."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True,
            gui_types="gui_1,gui_2",
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())

        assert "results" in summary
        assert "gui_1" in summary["results"] or len(summary["results"]) > 0


class TestGUIErrorHandling:
    """Tests for GUI error handling."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_nonexistent_target_directory(self, isolated_temp_dir: Any) -> Any:
        """Test handling of nonexistent target directory."""
        target = isolated_temp_dir / "nonexistent"
        output = isolated_temp_dir / "output"

        logger = get_real_logger()
        result = process_gui(
            target_dir=target, output_dir=output, logger=logger, headless=True
        )
        # Should handle gracefully (may succeed or fail based on implementation)
        assert isinstance(result, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_verbose_mode_logs_more(self, isolated_temp_dir: Any) -> Any:
        """Test that verbose mode produces more log messages."""
        target = isolated_temp_dir / "input"
        output_quiet = isolated_temp_dir / "output_quiet"
        output_verbose = isolated_temp_dir / "output_verbose"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger_quiet = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output_quiet,
            logger=logger_quiet,
            headless=True,
            verbose=False,
        )

        logger_verbose = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output_verbose,
            logger=logger_verbose,
            headless=True,
            verbose=True,
        )

        # Verbose should produce at least as many messages
        assert len(logger_verbose.get_messages()) >= len(logger_quiet.get_messages())


class TestGUI3DesignStudioActions:
    """Exercise GUI 3 callbacks without launching an optional web backend."""

    @pytest.mark.unit
    def test_state_space_buttons_mutate_list_rows(self) -> None:
        from gui.gui_3.ui_designer import (
            _add_state_space_row,
            _remove_last_state_space_row,
        )

        rows = [["A", "3,3", "Likelihood"]]
        added = _add_state_space_row(rows)
        assert added == [
            ["A", "3,3", "Likelihood"],
            ["x1", "1", "New state-space variable"],
        ]
        assert _remove_last_state_space_row(added) == rows
        assert _remove_last_state_space_row([]) == []

    @pytest.mark.unit
    def test_add_mapping_updates_existing_variable(self) -> None:
        from gui.gui_3.ui_designer import _add_ontology_mapping

        rows = [["A", "OldTerm", "Old description"]]
        updated = _add_ontology_mapping(rows, "A", "LikelihoodMatrix")
        assert updated == [["A", "LikelihoodMatrix", "Maps states to observations"]]
        assert _add_ontology_mapping(updated, "", "TransitionMatrix") == updated

    @pytest.mark.unit
    def test_connection_validation_and_layout_are_live_and_safe(self) -> None:
        from gui.gui_3.ui_designer import (
            _generate_connections_html,
            _validate_connections,
        )

        assert _validate_connections("D>s\ns-A\nA-o").startswith("✅ 3")
        assert _validate_connections("D => s").startswith("❌")
        graph = _generate_connections_html("D>s\ns-A\nA-o")
        assert "<svg" in graph
        assert "3 connection(s)" in graph
        escaped = _generate_connections_html("<img src=x onerror=alert(1)>")
        assert "<img" not in escaped
        assert "&lt;img" in escaped

    @pytest.mark.unit
    def test_list_backed_tables_export_complete_gnn(self) -> None:
        from gui.gui_3.ui_designer import _generate_gnn_from_design

        content = _generate_gnn_from_design(
            [
                ["A", "3,3,type=float", "Likelihood\nMatrix"],
                ["D", "3", "Prior"],
                ["s", "3", "Hidden state"],
                ["o", "3", "Observation"],
            ],
            [
                ["A", "LikelihoodMatrix", "Maps states to observations"],
                ["D", "PriorOverHiddenStates", "Initial beliefs"],
            ],
            "D>s\ns-A\nA-o",
            3,
            3,
            2,
            1,
            "Unbounded",
        )
        assert "A[3,3,type=float]" in content
        assert "type=float,type=float" not in content
        assert "Likelihood Matrix" in content
        assert "D[3,type=float]" in content
        assert "A=LikelihoodMatrix" in content
        assert "num_actions: 2" in content

    @pytest.mark.unit
    def test_loaded_model_parameters_seed_default_controls(self) -> None:
        from gui.gui_3.ui_designer import _parse_gnn_for_design

        parsed = _parse_gnn_for_design(
            "## StateSpaceBlock\nA[4,2,type=float]\n"
            "## Connections\nA>o\n"
            "## ModelParameters\nnum_hidden_states: 4\nplanning_horizon: 2\n"
        )
        assert parsed["state_spaces"] == [["A", "4,2", ""]]
        assert parsed["parameters"] == {
            "num_hidden_states": "4",
            "planning_horizon": "2",
        }

    @pytest.mark.unit
    def test_export_rejects_undefined_connection_variables(self) -> None:
        from gui.gui_3.ui_designer import _generate_gnn_from_design

        with pytest.raises(ValueError, match="undefined variable"):
            _generate_gnn_from_design(
                [["A", "3", "Likelihood"]],
                [],
                "A>missing",
                3,
                3,
                3,
                1,
                "Bounded",
            )

    @pytest.mark.unit
    def test_export_supports_unicode_model_variables(self) -> None:
        from gui.gui_3.ui_designer import _generate_gnn_from_design

        content = _generate_gnn_from_design(
            [["π", "3", "Policy"], ["G", "π,type=float", "Expected free energy"]],
            [],
            "π>G",
            3,
            3,
            3,
            1,
            "Bounded",
        )
        assert "π[3,type=float]" in content
        assert "G[π,type=float]" in content

    @pytest.mark.unit
    def test_every_design_studio_action_button_is_bound(self) -> None:
        """Prevent visible GUI controls from regressing to no-op buttons."""
        import inspect

        from gui.gui_3.ui_designer import build_design_studio

        source = inspect.getsource(build_design_studio)
        for button_name in (
            "add_variable_btn",
            "remove_variable_btn",
            "add_mapping_btn",
            "validate_connections_btn",
            "auto_layout_btn",
            "export_btn",
            "preview_btn",
        ):
            assert f"{button_name}.click(" in source
