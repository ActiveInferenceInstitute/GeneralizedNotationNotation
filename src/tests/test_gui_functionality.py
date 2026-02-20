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

import pytest
from pathlib import Path
import logging
import json

from gui import process_gui, generate_html_navigation


import io

def get_real_logger():
    """Create a real logger that captures output to a StringIO stream."""
    logger = logging.getLogger("test_gui_logger")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicate logs in parametrized tests
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
    logger.addHandler(handler)
    
    logger.stream = stream
    
    # Helper to retrieve trapped messages exactly like the mock
    def get_messages(level=None):
        content = stream.getvalue().splitlines()
        if not level:
            return [line.split(":", 1)[1] for line in content if ":" in line]
        level_str = level.upper()
        return [line.split(":", 1)[1] for line in content if line.startswith(f"{level_str}:")]
    
    logger.get_messages = get_messages
    return logger


class TestGUIHeadlessMode:
    """Tests for GUI headless mode execution."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_mode_returns_success(self, isolated_temp_dir):
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
            headless=True
        )
        assert result is True

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_mode_creates_summary(self, isolated_temp_dir):
        """Test that headless mode creates processing summary."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True
        )

        summary_file = output / "gui_processing_summary.json"
        assert summary_file.exists()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_summary_has_correct_mode(self, isolated_temp_dir):
        """Test that processing summary reports correct mode."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())
        assert summary.get("mode") == "headless"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_headless_mode_empty_directory(self, isolated_temp_dir):
        """Test headless mode with empty input directory."""
        target = isolated_temp_dir / "empty_input"
        output = isolated_temp_dir / "output"
        target.mkdir(parents=True, exist_ok=True)

        logger = get_real_logger()
        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True
        )
        # Should handle empty directory gracefully
        assert isinstance(result, bool)


class TestGUIConfiguration:
    """Tests for GUI configuration options."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui_types_string_parsing(self, isolated_temp_dir):
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
            gui_types="gui_1"
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())
        assert "gui_1" in summary.get("gui_types", [])

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui_types_list(self, isolated_temp_dir):
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
            gui_types=["gui_1", "gui_2"]
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())
        assert "gui_1" in summary.get("gui_types", [])
        assert "gui_2" in summary.get("gui_types", [])

    @pytest.mark.unit
    @pytest.mark.fast
    def test_unknown_gui_type_handled(self, isolated_temp_dir, caplog):
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
                gui_types="nonexistent_gui"
            )

        # Should have warning about unknown GUI type in captured logs
        assert any("Unknown" in r.message or "nonexistent" in r.message
                   for r in caplog.records if r.levelno >= logging.WARNING)


class TestGUIHTMLNavigation:
    """Tests for HTML navigation generation."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_generate_html_navigation_creates_file(self, isolated_temp_dir):
        """Test that HTML navigation file is created."""
        pipeline_output = isolated_temp_dir / "output"
        gui_output = pipeline_output / "22_gui_output"
        gui_output.mkdir(parents=True, exist_ok=True)

        # Create some dummy output directories
        (pipeline_output / "3_gnn_output").mkdir()
        (pipeline_output / "3_gnn_output" / "test.json").write_text("{}")

        logger = get_real_logger()
        result = generate_html_navigation(pipeline_output, gui_output, logger)

        nav_file = gui_output / "navigation.html"
        assert nav_file.exists()
        assert result is True

    @pytest.mark.unit
    @pytest.mark.fast
    def test_html_navigation_contains_structure(self, isolated_temp_dir):
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
    def test_html_navigation_empty_output(self, isolated_temp_dir):
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
    def test_process_gui_creates_output_directory(self, isolated_temp_dir):
        """Test that process_gui creates output directory if needed."""
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "nonexistent" / "output"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.md").write_text("# Test Model\n")

        logger = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True
        )

        assert output.exists()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_summary_contains_results(self, isolated_temp_dir):
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
            gui_types="gui_1,gui_2"
        )

        summary_file = output / "gui_processing_summary.json"
        summary = json.loads(summary_file.read_text())

        assert "results" in summary
        assert "gui_1" in summary["results"] or len(summary["results"]) > 0


class TestGUIErrorHandling:
    """Tests for GUI error handling."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_nonexistent_target_directory(self, isolated_temp_dir):
        """Test handling of nonexistent target directory."""
        target = isolated_temp_dir / "nonexistent"
        output = isolated_temp_dir / "output"

        logger = get_real_logger()
        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=logger,
            headless=True
        )
        # Should handle gracefully (may succeed or fail based on implementation)
        assert isinstance(result, bool)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_verbose_mode_logs_more(self, isolated_temp_dir):
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
            verbose=False
        )

        logger_verbose = get_real_logger()
        process_gui(
            target_dir=target,
            output_dir=output_verbose,
            logger=logger_verbose,
            headless=True,
            verbose=True
        )

        # Verbose should produce at least as many messages
        assert len(logger_verbose.get_messages()) >= len(logger_quiet.get_messages())
