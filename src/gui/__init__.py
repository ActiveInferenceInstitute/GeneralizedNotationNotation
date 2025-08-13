"""
GUI module for the Interactive GNN Constructor.

Public API:
- run_gui: launch or generate GUI artifacts
- add_component_to_markdown: add a component block to markdown
- update_component_states: append or replace states for a component
- remove_component_from_markdown: remove a component block by name
- parse_components_from_markdown: parse components section to structured list
"""

from .processor import run_gui
from .markdown import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
)

__all__ = [
    "run_gui",
    "add_component_to_markdown",
    "update_component_states",
    "remove_component_from_markdown",
    "parse_components_from_markdown",
    "parse_state_space_from_markdown",
    "add_state_space_entry",
    "update_state_space_entry",
    "remove_state_space_entry",
]


