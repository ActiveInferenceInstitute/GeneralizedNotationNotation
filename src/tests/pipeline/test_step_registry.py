#!/usr/bin/env python3
"""Tests for the canonical pipeline step registry (src/pipeline/step_registry.py)."""

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from pipeline.step_registry import (  # noqa: E402
    PIPELINE_STEPS_TUPLE,
    STANDARD_MODULE_FUNCTION_NAMES,
    STEP_METADATA_DICT,
    STEPS,
    StepInfo,
    get_core_steps,
    get_llm_steps,
    get_step_tags,
    output_dir_for_stem,
    step_for_name,
    step_for_stem,
)


class TestStepRegistryStructural:
    """Structural invariants of the registry itself."""

    def test_exactly_25_steps(self) -> None:
        """The pipeline has exactly 25 steps (0-24)."""
        assert len(STEPS) == 25

    def test_all_steps_have_unique_stems(self) -> None:
        """No duplicate script stems."""
        stems = [s.script_stem for s in STEPS]
        assert len(stems) == len(set(stems))

    def test_all_steps_have_required_fields(self) -> None:
        """Every StepInfo has non-empty script_stem, description, module_function."""
        for step in STEPS:
            assert step.script_stem, f"Empty script_stem at {step}"
            assert step.description, f"Empty description at {step}"
            assert step.module_function, f"Empty module_function at {step}"
            assert isinstance(step.tags, frozenset), f"tags not frozenset at {step}"

    def test_script_name_format(self) -> None:
        """Every script_name follows <N>_<name>.py."""
        for step in STEPS:
            name = step.script_name
            assert name.endswith(".py"), f"Missing .py: {name}"
            stem = name[:-3]
            parts = stem.split("_", 1)
            assert parts[0].isdigit(), f"Non-numeric prefix: {name}"

    def test_output_dir_format(self) -> None:
        """Every output_dir_name follows <stem>_output."""
        for step in STEPS:
            assert step.output_dir_name.endswith("_output"), (
                f"Bad output dir: {step.output_dir_name}"
            )
            assert step.output_dir_name.startswith(step.script_stem), (
                f"Mismatch: {step.output_dir_name} vs {step.script_stem}"
            )

    def test_module_function_resolves_on_all_steps(self) -> None:
        """Every step's module_function is a real callable on its numbered script.

        Supersedes the former prefix-only "plausibility" check: metadata is only
        valid when the named function actually exists and is callable on the
        script module (enforced by
        ``test_module_functions_resolve_on_numbered_scripts``).
        """
        assert all(step.module_function for step in STEPS)


class TestStepRegistryLookups:
    """Lookup functions return correct results."""

    def test_step_for_stem_exists(self) -> None:
        """step_for_stem returns StepInfo for valid stems."""
        s = step_for_stem("11_render")
        assert s is not None
        assert s.script_name == "11_render.py"
        assert "core" in s.tags

    def test_step_for_name_accepts_script(self) -> None:
        """step_for_name works with both script_name and stem."""
        s1 = step_for_name("11_render.py")
        s2 = step_for_name("11_render")
        assert s1 is not None and s2 is not None
        assert s1 == s2

    def test_step_for_nonexistent(self) -> None:
        """Step lookup returns None for unknown stems."""
        assert step_for_stem("99_nonexistent") is None
        assert step_for_name("99_nonexistent.py") is None

    def test_llm_step_is_13(self) -> None:
        """Only step 13 has the 'llm' tag."""
        llm_steps = get_llm_steps()
        assert len(llm_steps) == 1
        assert llm_steps[0].script_stem == "13_llm"

    def test_core_steps_exclude_llm(self) -> None:
        """Core steps exclude step 13 (LLM)."""
        core_steps = get_core_steps()
        core_stems = {s.script_stem for s in core_steps}
        assert "13_llm" not in core_stems

    def test_output_dir_for_stem(self) -> None:
        """output_dir_for_stem returns expected directory names."""
        assert output_dir_for_stem("3_gnn") == "3_gnn_output"
        assert output_dir_for_stem("11_render") == "11_render_output"
        assert output_dir_for_stem("nonexistent") is None

    def test_get_step_tags(self) -> None:
        """get_step_tags returns tags for valid stems, empty for invalid."""
        assert "core" in get_step_tags("3_gnn")
        assert "llm" in get_step_tags("13_llm")
        assert get_step_tags("nonexistent") == frozenset()

    def test_get_stage_steps(self) -> None:
        """get_stage_steps returns steps grouped by logical stage."""
        from pipeline.step_registry import get_stage_steps

        discovery = get_stage_steps("discovery_schema")
        assert len(discovery) == 7
        assert discovery[0].script_stem == "0_template"
        assert discovery[-1].script_stem == "6_validation"

        sim = get_stage_steps("simulation_execution")
        assert len(sim) == 4
        assert {s.script_stem for s in sim} == {
            "11_render",
            "12_execute",
            "15_audio",
            "16_analysis",
        }

    def test_consolidated_step_aliases(self) -> None:
        """Test step alias resolution for renumbering migration."""
        from pipeline.step_registry import (
            canonical_step_stem,
            step_for_name,
            step_for_stem,
        )

        assert canonical_step_stem("13_audio") == "15_audio"
        assert canonical_step_stem("14_analysis") == "16_analysis"
        assert canonical_step_stem("11_render") == "11_render"

        # Check step_for_stem and step_for_name resolution with aliases
        s_audio = step_for_stem("13_audio")
        assert s_audio is not None
        assert s_audio.script_stem == "15_audio"

        s_audio_py = step_for_name("13_audio.py")
        assert s_audio_py is not None
        assert s_audio_py.script_name == "15_audio.py"


class TestDerivedExportAliases:
    """Derived-export aliases match STEPS."""

    def test_pipeline_steps_tuple(self) -> None:
        """PIPELINE_STEPS_TUPLE mirrors STEPS 1:1."""
        assert len(PIPELINE_STEPS_TUPLE) == len(STEPS)
        for i, (script, desc) in enumerate(PIPELINE_STEPS_TUPLE):
            assert script == STEPS[i].script_name
            assert desc == STEPS[i].description

    def test_step_metadata_dict(self) -> None:
        """STEP_METADATA_DICT has all stems as keys."""
        assert set(STEP_METADATA_DICT.keys()) == {s.script_stem for s in STEPS}
        for stem, meta in STEP_METADATA_DICT.items():
            assert "name" in meta
            assert "description" in meta

    def test_standard_module_function_names(self) -> None:
        """STANDARD_MODULE_FUNCTION_NAMES maps all stems to functions."""
        assert set(STANDARD_MODULE_FUNCTION_NAMES.keys()) == {
            s.script_stem for s in STEPS
        }
        for stem, func in STANDARD_MODULE_FUNCTION_NAMES.items():
            step = step_for_stem(stem)
            assert step is not None
            assert func == step.module_function, (
                f"Function mismatch for {stem}: {func} vs {step.module_function}"
            )

    def test_module_functions_resolve_on_numbered_scripts(self) -> None:
        """Registry module_function names resolve to callables on the scripts.

        The registry is the canonical step list; its ``module_function``
        metadata must match what each numbered script actually delegates to
        (the callable passed to ``create_standardized_pipeline_script``).
        """
        import importlib

        for step in STEPS:
            module = importlib.import_module(step.script_stem)
            func = getattr(module, step.module_function, None)
            assert callable(func), (
                f"{step.script_stem}: registry function {step.module_function!r} "
                "is not callable on the script module"
            )


class TestStepInfoDataclass:
    """StepInfo frozen dataclass behavior."""

    def test_step_info_is_frozen(self) -> None:
        """StepInfo instances cannot be mutated."""
        s = STEPS[0]
        with pytest.raises((AttributeError, TypeError)):
            s.script_stem = "changed"  # type: ignore

    def test_step_info_equality(self) -> None:
        """Two StepInfo with same fields are equal."""
        a = StepInfo("test", "testing", "process_test", frozenset({"core"}))
        b = StepInfo("test", "testing", "process_test", frozenset({"core"}))
        assert a == b

    def test_step_info_hashable(self) -> None:
        """StepInfo is hashable (frozen dataclass)."""
        s = STEPS[0]
        d: dict[Any, Any] = {s: "found"}
        assert d[s] == "found"
