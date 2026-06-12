---
name: gnn-gui-builder
description: GNN interactive GUI for constructing and editing GNN models. Use when building visual model editors, launching the GNN GUI application, or working with the multi-panel GUI system (gui_1, gui_2, gui_3, oxdraw).
---

# GNN GUI Builder (Step 22)

## Purpose

Provides interactive graphical interfaces for constructing, editing, and visualizing GNN models. Includes multiple GUI implementations (gui_1, gui_2, gui_3, oxdraw) with increasing capability levels.

## Key Commands

```bash
# Launch GUI
python src/22_gui.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 22 --verbose
```

## API

```python
from gui import (
    process_gui, get_available_guis,
    gui_1, gui_2, gui_3, oxdraw_gui,
    get_gui_1_info, get_gui_2_info, get_gui_3_info, get_oxdraw_info,
    generate_html_navigation,
    parse_components_from_markdown, parse_state_space_from_markdown,
    add_component_to_markdown, update_component_states,
    add_state_space_entry, update_state_space_entry
)

# Process GUI step (used by pipeline)
process_gui(target_dir, output_dir, verbose=True)

# Get available GUIs
guis = get_available_guis()

# Launch specific GUI
gui_1()   # Basic
gui_2()   # Enhanced
gui_3()   # Full-featured
oxdraw_gui()  # Graph drawing canvas

# Parse GNN markdown for GUI
components = parse_components_from_markdown(md_content)
state_space = parse_state_space_from_markdown(md_content)
```

## Key Exports

- `process_gui` — main pipeline processing function
- `gui_1`, `gui_2`, `gui_3`, `oxdraw_gui` — GUI launchers
- `get_available_guis` — list available GUI implementations
- `parse_components_from_markdown`, `parse_state_space_from_markdown` — content parsing
- `add_component_to_markdown`, `update_component_states` — markdown manipulation

## Dependencies

```bash
uv sync --extra gui
```

## Output

- GUI session data in `output/22_gui_output/`
- Saved model files from GUI editing sessions


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_gui_module_info`
- `list_available_guis`
- `process_gui`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
