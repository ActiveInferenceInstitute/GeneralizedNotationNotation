"""v4.0.0 retirement pins: the retired ``validate_gnn*`` aliases are gone.

The MAJ-05 window (opened 2026-09-11) closed in the v4.0.0 cycle: all ten
old-name alias defs were deleted and their re-export wiring pruned. Every
retired name must now be absent from its owning module — re-adding an alias
(with or without a warning) must fail here.
"""

from __future__ import annotations

import importlib

import pytest


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
