"""Static 25-step site catalogue for the website generator.

Leaf module: owns ``StepInfo``, ``PIPELINE_STEPS``, and
``get_pipeline_steps`` so both ``generator.py`` and ``collection.py``
import the catalogue from here instead of importing each other — the
generator↔collection import cycle is broken here (steps.py depends only
on the canonical ``gnn.pipeline.step_registry``).
"""

from dataclasses import dataclass

from gnn.pipeline.step_registry import STEPS as _REGISTRY_STEPS

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline step catalogue
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepInfo:
    """One pipeline step in the static 25-step site catalogue."""

    number: int
    name: str
    description: str

    @property
    def script_name(self) -> str:
        """Conventional display name of the numbered orchestrator script."""
        return f"{self.number}_{self.name.lower().replace(' ', '_')}.py"

    @property
    def output_dir_name(self) -> str:
        """Standard output subdirectory, mirroring the step registry (``11_render_output``)."""
        return f"{self.number}_{self.name.lower().replace(' ', '_')}_output"


# Acronym casing for stem suffixes that must not be title-cased
# (``"mcp".title()`` would yield "Mcp", breaking the display name and the
# ``script_name`` round-trip back to the real orchestrator script stem).
_ACRONYM_DISPLAY: dict[str, str] = {
    "gnn": "GNN",
    "gui": "GUI",
    "llm": "LLM",
    "mcp": "MCP",
    "ml": "ML",
}


def _display_name_from_stem_suffix(suffix: str) -> str:
    """Display name for a registry stem suffix (``"advanced_viz"`` → ``"Advanced Viz"``)."""
    return " ".join(
        _ACRONYM_DISPLAY.get(word, word.title()) for word in suffix.split("_")
    )


def _steps_from_registry() -> tuple[StepInfo, ...]:
    """Derive the site catalogue from the canonical ``step_registry.STEPS``."""
    infos: list[StepInfo] = []
    for registry_step in _REGISTRY_STEPS:
        number_str, _, suffix = registry_step.script_stem.partition("_")
        infos.append(
            StepInfo(
                number=int(number_str),
                name=_display_name_from_stem_suffix(suffix),
                description=registry_step.description,
            )
        )
    return tuple(infos)


PIPELINE_STEPS: tuple[StepInfo, ...] = _steps_from_registry()


def get_pipeline_steps() -> tuple[StepInfo, ...]:
    """Return the immutable 25-step catalogue rendered across the site."""
    return PIPELINE_STEPS
