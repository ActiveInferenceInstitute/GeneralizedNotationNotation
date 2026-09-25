"""Tests for the registry-derived ``PIPELINE_OUTPUT_SECTIONS`` catalogue.

Pins that the GUI navigation's step table derives from the canonical
``gnn.pipeline.step_registry`` (count, order, output dirs), reproduces the
established 25-entry table byte-for-byte, and extends automatically when a
new step joins the registry — the drift the derivation removes.
"""

from __future__ import annotations

import pytest

from gnn.gui.processor import (
    _ACRONYM_DISPLAY,
    _SECTION_PATTERNS,
    _SECTION_TITLE_OVERRIDES,
    PIPELINE_OUTPUT_SECTIONS,
    _display_name_from_stem,
    derive_pipeline_output_sections,
)
from gnn.pipeline.step_registry import STEPS, StepInfo

EXPECTED_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Template", "0_template_output", ("*.json", "*.md")),
    ("Setup", "1_setup_output", ("*.json",)),
    ("Tests", "2_tests_output", ("*.txt", "*.json")),
    ("GNN Processing", "3_gnn_output", ("*.json", "*.md", "*.pkl")),
    ("Model Registry", "4_model_registry_output", ("*.json",)),
    ("Type Checker", "5_type_checker_output", ("*.json", "*.md")),
    ("Validation", "6_validation_output", ("*.json",)),
    ("Export", "7_export_output", ("*.json", "*.xml", "*.pkl")),
    ("Visualization", "8_visualization_output", ("*.png", "*.svg", "*.csv", "*.json")),
    ("Advanced Visualization", "9_advanced_viz_output", ("*.png", "*.json")),
    ("Ontology", "10_ontology_output", ("*.json",)),
    ("Render", "11_render_output", ("*.py", "*.jl", "*.md", "*.json", "*.png")),
    ("Execute", "12_execute_output", ("*.txt", "*.json", "*.md", "*.png")),
    ("LLM", "13_llm_output", ("*.md", "*.json")),
    ("ML Integration", "14_ml_integration_output", ("*.json",)),
    ("Audio", "15_audio_output", ("*.json", "*.wav")),
    ("Analysis", "16_analysis_output", ("*.json",)),
    ("Integration", "17_integration_output", ("*.json",)),
    ("Security", "18_security_output", ("*.json",)),
    ("Research", "19_research_output", ("*.json",)),
    ("Website", "20_website_output", ("*.html", "*.json")),
    ("MCP", "21_mcp_output", ("*.json",)),
    ("GUI", "22_gui_output", ("*.md", "*.json")),
    ("Report", "23_report_output", ("*.html", "*.md", "*.json")),
    (
        "Intelligent Analysis",
        "24_intelligent_analysis_output",
        ("*.json", "*.md", "*.html"),
    ),
)


class TestRegistryDerivation:
    """PIPELINE_OUTPUT_SECTIONS is registry-derived, not a second table."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_sections_count_matches_registry(self) -> None:
        assert len(PIPELINE_OUTPUT_SECTIONS) == len(STEPS)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_sections_dirs_match_registry_order(self) -> None:
        for (name, step_dir, patterns), step in zip(
            PIPELINE_OUTPUT_SECTIONS, STEPS, strict=True
        ):
            assert step_dir == step.output_dir_name
            assert name
            assert all(pattern.startswith("*.") for pattern in patterns)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_sections_reproduce_established_table(self) -> None:
        """The derivation reproduces the known-good 25-entry table exactly."""
        assert PIPELINE_OUTPUT_SECTIONS == EXPECTED_SECTIONS

    @pytest.mark.unit
    @pytest.mark.fast
    def test_new_registry_step_flows_through(self) -> None:
        extended = derive_pipeline_output_sections(
            [*STEPS, StepInfo("25_demo", "Demo step", "process_demo")]
        )
        assert len(extended) == len(STEPS) + 1
        assert extended[-1] == (
            "Demo",
            "25_demo_output",
            ("*.json", "*.md"),
        )

    @pytest.mark.unit
    @pytest.mark.fast
    def test_pattern_map_keys_are_live_registry_stems(self) -> None:
        stems = {step.script_stem for step in STEPS}
        assert set(_SECTION_PATTERNS) <= stems
        suffixes = {stem.partition("_")[2] for stem in stems}
        assert set(_SECTION_TITLE_OVERRIDES) <= suffixes
        assert set(_ACRONYM_DISPLAY) <= {
            word for stem in stems for word in stem.partition("_")[2].split("_")
        }
