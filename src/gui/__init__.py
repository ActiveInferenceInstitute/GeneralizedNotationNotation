"""
GUI module for Interactive GNN Constructors.

This module provides multiple GUI implementations:
- GUI 1: Form-based Interactive GNN Constructor
- GUI 2: Visual Matrix Editor with drag-and-drop
- GUI 3: State Space Design Studio
- oxdraw: Visual diagram-as-code interface

Public API:
- process_gui: main processing function (runs all available GUIs)
- gui_1: form-based GUI with component management
- gui_2: visual matrix editor with drag-and-drop
- gui_3: state space design studio
- oxdraw: visual diagram-as-code with Mermaid
- get_available_guis: list all available GUI implementations
"""

__version__ = "1.1.3"
FEATURES = {
    "form_based_constructor": True,
    "visual_matrix_editor": True,
    "state_space_studio": True,
    "diagram_as_code": True,
    "mcp_integration": True
}

# Import GUI runners
from .gui_1 import gui_1, get_gui_1_info
from .gui_2 import gui_2, get_gui_2_info
from .gui_3 import gui_3, get_gui_3_info
from .oxdraw import oxdraw_gui, get_oxdraw_info

# Import GUI 1 utilities
from .gui_1 import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
)

# Import main processing functions from processor
from .processor import (
    process_gui,
    generate_html_navigation,
)


def get_available_guis():
    """Get list of available GUI implementations with their info"""
    return {
        "gui_1": get_gui_1_info(),
        "gui_2": get_gui_2_info(),
        "gui_3": get_gui_3_info(),
        "oxdraw": get_oxdraw_info(),
    }


__all__ = [
    "process_gui",
    "gui_1",
    "gui_2",
    "gui_3",
    "oxdraw_gui",
    "get_available_guis",
    "get_gui_1_info",
    "get_gui_2_info",
    "get_gui_3_info",
    "get_oxdraw_info",
    "generate_html_navigation",
    # GUI 1 utilities
    "add_component_to_markdown",
    "update_component_states",
    "remove_component_from_markdown",
    "parse_components_from_markdown",
    "parse_state_space_from_markdown",
    "add_state_space_entry",
    "update_state_space_entry",
    "remove_state_space_entry",
]
