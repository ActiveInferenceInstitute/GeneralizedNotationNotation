"""Tests for the static step catalogue in ``gnn.website.steps``.

Verifies that the 25-step site catalogue is derived from the canonical
``gnn.pipeline.step_registry.STEPS``, exposes the shared dataclass and
constants via the package, and keeps its per-step derived accessors
(``script_name``, ``output_dir_name``) consistent with the registry.
"""

import dataclasses

import pytest

import gnn.website
import gnn.website.renderer
import gnn.website.steps
from gnn.pipeline.step_registry import STEPS as REGISTRY_STEPS
from gnn.website import PIPELINE_STEPS, StepInfo, get_pipeline_steps
from gnn.website.steps import (
    _ACRONYM_DISPLAY,
    _display_name_from_stem_suffix,
    _steps_from_registry,
)


class TestStepInfo:
    """Unit behavior of the ``StepInfo`` dataclass and its derived properties."""

    def test_is_frozen_dataclass(self):
        assert issubclass(StepInfo, object)
        step = StepInfo(number=1, name="Load", description="load model files")
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.number = 2  # frozen dataclass must reject mutation

    def test_fields(self):
        step = StepInfo(number=7, name="Type Check", description="validate types")
        assert step.number == 7
        assert step.name == "Type Check"
        assert step.description == "validate types"

    def test_script_name(self):
        step = StepInfo(number=11, name="Render Output", description="render")
        assert step.script_name == "11_render_output.py"

    def test_output_dir_name(self):
        step = StepInfo(number=11, name="Render Output", description="render")
        assert step.output_dir_name == "11_render_output_output"


class TestCatalogueFromRegistry:
    """The catalogue must mirror the canonical step registry."""

    def test_matches_registry_count(self):
        assert len(PIPELINE_STEPS) == len(REGISTRY_STEPS)
        assert len(PIPELINE_STEPS) == 25

    def test_matches_registry_numbers_and_stems(self):
        for step, registry_step in zip(PIPELINE_STEPS, REGISTRY_STEPS, strict=True):
            number_str, _, suffix = registry_step.script_stem.partition("_")
            assert step.number == int(number_str)
            assert step.name.lower().replace(" ", "_") == suffix

    def test_matches_registry_descriptions(self):
        for step, registry_step in zip(PIPELINE_STEPS, REGISTRY_STEPS, strict=True):
            assert step.description == registry_step.description

    def test_script_name_round_trips_to_registry_stem(self):
        for step, registry_step in zip(PIPELINE_STEPS, REGISTRY_STEPS, strict=True):
            assert step.script_name == f"{registry_step.script_stem}.py"

    def test_output_dir_name_matches_registry(self):
        for step, registry_step in zip(PIPELINE_STEPS, REGISTRY_STEPS, strict=True):
            assert step.output_dir_name == f"{registry_step.script_stem}_output"

    def test_numbers_strictly_increasing(self):
        numbers = [step.number for step in PIPELINE_STEPS]
        assert numbers == sorted(numbers)
        assert len(set(numbers)) == len(numbers)

    def test_get_pipeline_steps_returns_catalogue(self):
        steps = get_pipeline_steps()
        assert isinstance(steps, tuple)
        assert steps == PIPELINE_STEPS

    def test_steps_from_registry_rebuilds_catalogue(self):
        assert _steps_from_registry() == PIPELINE_STEPS


class TestDisplayNames:
    """Acronym casing and title-casing of stem suffixes."""

    def test_acronym_map_known_keys(self):
        for key in _ACRONYM_DISPLAY:
            assert key.islower()
        assert _ACRONYM_DISPLAY == {
            "gnn": "GNN",
            "gui": "GUI",
            "llm": "LLM",
            "mcp": "MCP",
            "ml": "ML",
        }

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            ("load", "Load"),
            ("type_check", "Type Check"),
            ("advanced_viz", "Advanced Viz"),
            ("mcp", "MCP"),
            ("gnn_model_list", "GNN Model List"),
            ("gui", "GUI"),
            ("llm", "LLM"),
            ("ml_training", "ML Training"),
        ],
    )
    def test_display_name_from_stem_suffix(self, suffix, expected):
        assert _display_name_from_stem_suffix(suffix) == expected

    def test_catalogue_names_match_display_name_derivation(self):
        for step, registry_step in zip(PIPELINE_STEPS, REGISTRY_STEPS, strict=True):
            _, _, suffix = registry_step.script_stem.partition("_")
            assert step.name == _display_name_from_stem_suffix(suffix)


class TestPackageReExports:
    """``gnn.website`` re-exports the catalogue shared with the renderer."""

    def test_reexports(self):
        assert gnn.website.PIPELINE_STEPS is gnn.website.steps.PIPELINE_STEPS
        assert gnn.website.StepInfo is gnn.website.steps.StepInfo
        assert gnn.website.get_pipeline_steps is gnn.website.steps.get_pipeline_steps

    def test_all_listed(self):
        for name in ("PIPELINE_STEPS", "StepInfo", "get_pipeline_steps"):
            assert name in gnn.website.__all__
