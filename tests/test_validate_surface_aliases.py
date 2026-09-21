"""v4.0.0 retirement pins: the deprecated ``validate_gnn*`` aliases are gone.

The MAJ-05 deprecation window (opened 2026-09-11) closed in the v4.0.0
cycle: all ten old-name alias defs were deleted and their re-export wiring
pruned. Every retired name must now be absent from its owning module —
re-adding an alias (with or without a warning) must fail here. The utils
``pipeline_template`` re-export pin lives here too (it rode along with the
MAJ-05 alias pins).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from gnn.pipeline.config import get_output_dir_for_script

RETIRED_ALIASES: list[tuple[str, str]] = [
    # (owning module, retired name)
    ("gnn", "validate_gnn_file"),
    ("gnn", "validate_gnn"),
    ("gnn", "validate_gnn_syntax_formal"),
    ("gnn", "validate_gnn_structure"),
    ("gnn.parsers", "validate_gnn"),
    ("gnn.parsers", "validate_gnn_syntax_formal"),
    ("gnn.parsers.basic", "validate_gnn"),
    ("gnn.parsers.basic", "validate_gnn_syntax_formal"),
    ("gnn.processing", "validate_gnn_structure"),
    ("gnn.processing.processor", "validate_gnn_structure"),
    ("gnn.execute.pymdp.pymdp_utils", "validate_gnn_pomdp_structure"),
    ("gnn.schema_validator", "validate_gnn_file"),
    ("gnn.schema_validator.validator", "validate_gnn_file"),
    ("gnn.validation.simple", "validate_gnn_file"),
    ("gnn.validation.simple", "validate_gnn_directory"),
    ("gnn.mcp.processors", "validate_gnn_cross_format_consistency"),
    ("gnn.llm.llm_operations", "validate_gnn"),
]


@pytest.mark.parametrize(("module_name", "alias"), RETIRED_ALIASES)
def test_retired_validate_gnn_aliases_are_gone(module_name: str, alias: str) -> None:
    """Importing or attribute-accessing a retired alias must fail."""
    module = importlib.import_module(module_name)
    assert not hasattr(module, alias), (
        f"{module_name}.{alias} was retired in the v4.0.0 cycle but is present"
    )


def test_pipeline_template_output_dir_reexport_warns(tmp_path: Path) -> None:
    """The legacy ``gnn.utils.pipeline_template`` re-export of the canonical
    ``gnn.pipeline.config.get_output_dir_for_script`` must warn and forward.
    """
    import gnn.utils.pipeline_orchestration.pipeline_template as template

    with pytest.warns(DeprecationWarning):
        legacy = template.get_output_dir_for_script  # noqa: B018
    assert callable(legacy)
    assert legacy("3_gnn.py", tmp_path) == get_output_dir_for_script(
        "3_gnn.py", tmp_path
    )
