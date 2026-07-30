#!/usr/bin/env python3
"""Canonical pipeline step registry — single source of truth for all 25 steps.

Every step list in the codebase should derive from ``STEPS`` below.
Add ONE entry here when adding a new pipeline step; all downstream lists
(main.py, config.py, pipeline_template.py, justfile) update automatically.

Usage:

    from pipeline.step_registry import STEPS, step_for_name, step_output_dir

    for step in STEPS:
        print(step.script_name, step.description)

    step = step_for_name("11_render")
    print(step.module_function)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StepInfo:
    """Immutable descriptor for one pipeline step."""

    script_stem: str  # e.g. "11_render"
    description: str  # e.g. "Code rendering"
    module_function: str  # e.g. "process_render"
    tags: frozenset[str] = frozenset(("core",))
    default_recursive: bool = False
    additional_args_key: str = ""  # key into STEP_ADDITIONAL_ARGUMENTS if any

    @property
    def script_name(self) -> str:
        """Full script filename, e.g. ``11_render.py``."""
        return f"{self.script_stem}.py"

    @property
    def output_dir_name(self) -> str:
        """Standard output subdirectory, e.g. ``11_render_output``."""
        return f"{self.script_stem}_output"


# ---------------------------------------------------------------------------
# Canonical step list — ordered, authoritative
# ---------------------------------------------------------------------------
STEPS: List[StepInfo] = [
    StepInfo("0_template", "Template initialization", "process_template", frozenset({"core"})),
    StepInfo("1_setup", "Environment setup", "process_setup", frozenset({"core"}),
             additional_args_key="1_setup"),
    StepInfo("2_tests", "Test suite execution", "run_tests", frozenset({"core", "tests"})),
    StepInfo("3_gnn", "GNN file processing", "process_gnn_files", frozenset({"core"})),
    StepInfo("4_model_registry", "Model registry", "process_model_registry", frozenset({"core"})),
    StepInfo("5_type_checker", "Type checking", "process_type_checking", frozenset({"core"}),
             additional_args_key="5_type_checker"),
    StepInfo("6_validation", "Validation", "process_validation", frozenset({"core"})),
    StepInfo("7_export", "Multi-format export", "process_export", frozenset({"core"})),
    StepInfo("8_visualization", "Visualization", "process_visualization", frozenset({"core"})),
    StepInfo("9_advanced_viz", "Advanced visualization", "process_advanced_viz", frozenset({"core"})),
    StepInfo("10_ontology", "Ontology processing", "process_ontology", frozenset({"core"}),
             additional_args_key="10_ontology"),
    StepInfo("11_render", "Code rendering", "process_render", frozenset({"core"})),
    StepInfo("12_execute", "Execution", "process_execute", frozenset({"core"})),
    StepInfo("13_llm", "LLM processing", "process_llm", frozenset({"llm"}),
             additional_args_key="13_llm"),
    StepInfo("14_ml_integration", "ML integration", "process_ml_integration", frozenset({"core"})),
    StepInfo("15_audio", "Audio processing", "process_audio", frozenset({"core"})),
    StepInfo("16_analysis", "Analysis", "process_analysis", frozenset({"core"})),
    StepInfo("17_integration", "Integration", "process_integration", frozenset({"core"})),
    StepInfo("18_security", "Security", "process_security", frozenset({"core"})),
    StepInfo("19_research", "Research", "process_research", frozenset({"core"})),
    StepInfo("20_website", "Website generation", "process_website", frozenset({"core"}),
             additional_args_key="20_website"),
    StepInfo("21_mcp", "Model Context Protocol processing", "process_mcp", frozenset({"core"}),
             additional_args_key="21_mcp"),
    StepInfo("22_gui", "GUI (Interactive GNN Constructor)", "process_gui", frozenset({"core"})),
    StepInfo("23_report", "Report generation", "process_report", frozenset({"core"}),
             additional_args_key="23_report"),
    StepInfo("24_intelligent_analysis", "Intelligent pipeline analysis",
             "process_intelligent_analysis", frozenset({"core"})),
]

# ---------------------------------------------------------------------------
# Derived lookup maps
# ---------------------------------------------------------------------------
_STEPS_BY_STEM: Dict[str, StepInfo] = {s.script_stem: s for s in STEPS}
_STEPS_BY_SCRIPT: Dict[str, StepInfo] = {s.script_name: s for s in STEPS}


def step_for_stem(stem: str) -> Optional[StepInfo]:
    """Look up a step by its script stem (e.g. ``\"11_render\"``)."""
    return _STEPS_BY_STEM.get(stem)


def step_for_name(name: str) -> Optional[StepInfo]:
    """Look up a step by script name (e.g. ``\"11_render.py\"``) or stem."""
    if name.endswith(".py"):
        return _STEPS_BY_SCRIPT.get(name)
    return _STEPS_BY_STEM.get(name)


def output_dir_for_stem(stem: str) -> Optional[str]:
    """Return the standard output dir name for a step stem, or None."""
    step = step_for_stem(stem)
    return step.output_dir_name if step else None


# ---------------------------------------------------------------------------
# Re-export aliases for existing code and tests
# ---------------------------------------------------------------------------
PIPELINE_STEPS_TUPLE = tuple(
    (step.script_name, step.description) for step in STEPS
)

STEP_METADATA_DICT: Dict[str, Dict[str, str]] = {
    step.script_stem: {"name": step.description, "description": step.description}
    for step in STEPS
}

STEP_OUTPUT_DIR_MAP: Dict[str, str] = {
    step.script_stem: step.output_dir_name for step in STEPS
}

STANDARD_MODULE_FUNCTION_NAMES: Dict[str, str] = {
    step.script_stem: step.module_function for step in STEPS
}


def get_step_tags(stem: str) -> frozenset[str]:
    """Return the tags for a given step stem (for pipeline filtering)."""
    step = step_for_stem(stem)
    return step.tags if step else frozenset()


def get_core_steps() -> List[StepInfo]:
    """Return only core-tagged steps (excludes LLM-only steps)."""
    return [s for s in STEPS if "core" in s.tags]


def get_llm_steps() -> List[StepInfo]:
    """Return only LLM-tagged steps."""
    return [s for s in STEPS if "llm" in s.tags]


def discover_steps() -> Dict[int, StepInfo]:
    """Return a dict mapping step number (from script_stem) to StepInfo."""
    return {int(s.script_stem.split("_")[0]): s for s in STEPS}
