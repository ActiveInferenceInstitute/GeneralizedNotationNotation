"""Shared pipeline step dependency definitions.

This module captures execution-order dependencies between numbered pipeline
steps. It intentionally models step-output prerequisites, not import-time source
dependencies such as Step 21 discovering module ``mcp.py`` files.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Iterable

PIPELINE_STEP_SCRIPTS = MappingProxyType(
    {
        0: "0_template.py",
        1: "1_setup.py",
        2: "2_tests.py",
        3: "3_gnn.py",
        4: "4_model_registry.py",
        5: "5_type_checker.py",
        6: "6_validation.py",
        7: "7_export.py",
        8: "8_visualization.py",
        9: "9_advanced_viz.py",
        10: "10_ontology.py",
        11: "11_render.py",
        12: "12_execute.py",
        13: "13_llm.py",
        14: "14_ml_integration.py",
        15: "15_audio.py",
        16: "16_analysis.py",
        17: "17_integration.py",
        18: "18_security.py",
        19: "19_research.py",
        20: "20_website.py",
        21: "21_mcp.py",
        22: "22_gui.py",
        23: "23_report.py",
        24: "24_intelligent_analysis.py",
    }
)

PIPELINE_SCRIPT_STEPS = MappingProxyType(
    {script: step for step, script in PIPELINE_STEP_SCRIPTS.items()}
)

PIPELINE_STEP_DEPENDENCIES = MappingProxyType(
    {
        0: (),
        1: (),
        2: (1,),
        3: (),
        4: (3,),
        5: (3,),
        6: (3, 5),
        7: (3,),
        8: (3,),
        9: (3, 8),
        10: (3,),
        11: (3,),
        12: (3, 11),
        13: (3,),
        14: (3,),
        15: (3,),
        16: (3, 7),
        17: tuple(range(3, 17)),
        18: (11,),
        19: (3,),
        20: (8,),
        # Step 21 discovers module MCP source files; it has no step-output prerequisite.
        21: (),
        22: (3,),
        23: (8, 13),
        24: tuple(range(0, 24)),
    }
)


def normalize_script_name(script_name: str | Path) -> str:
    """Return a canonical script filename for a step identifier."""
    return Path(str(script_name)).name


def step_number_for_script(script_name: str | Path) -> int | None:
    """Return the numeric step for a script filename, if known."""
    return PIPELINE_SCRIPT_STEPS.get(normalize_script_name(script_name))


def dependency_steps_for_step(step_number: int) -> tuple[int, ...]:
    """Return direct dependency step numbers for ``step_number``."""
    return tuple(PIPELINE_STEP_DEPENDENCIES.get(step_number, ()))


def dependency_scripts_for_script(script_name: str | Path) -> list[str]:
    """Return direct dependency script filenames for ``script_name``."""
    step_number = step_number_for_script(script_name)
    if step_number is None:
        return []
    return [
        PIPELINE_STEP_SCRIPTS[dep]
        for dep in dependency_steps_for_step(step_number)
        if dep in PIPELINE_STEP_SCRIPTS
    ]


def resolve_step_dependencies(requested_steps: Iterable[int]) -> list[int]:
    """Return requested steps plus recursive dependencies in pipeline order."""
    resolved: set[int] = set()
    visiting: set[int] = set()

    def visit(step_number: int) -> None:
        """Provide visit behavior."""
        if step_number in resolved:
            return
        if step_number in visiting:
            raise ValueError(
                f"Cycle detected in pipeline dependencies at step {step_number}"
            )
        visiting.add(step_number)
        for dependency in dependency_steps_for_step(step_number):
            visit(dependency)
        visiting.remove(step_number)
        resolved.add(step_number)

    for step_number in requested_steps:
        visit(step_number)

    return sorted(resolved)
