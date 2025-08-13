# GUI Module (Interactive GNN Constructor)

## Overview
- Interactive two-pane editor for GNN models.
- Left panel (Controls):
  - Components tab: add/remove components, set type (observation/hidden/action/policy), and manage states
  - State Space tab: live list of state-space entries with name, dimensions, type; add/update/remove entries
- Right panel: synchronized plaintext GNN markdown editor
- Edits are immediately reflected in the markdown; save writes to a file in the step output directory

## Usage
```bash
python src/22_gui.py --target-dir input/gnn_files --output-dir output --verbose
```

- With Gradio available, a local web UI launches (in-browser). Otherwise, headless artifacts are generated.

## Headless Mode
- If the GUI cannot be launched (or headless=True):
  - gui_status.json summarizes backend availability and export path
  - constructed_model.md is created as a starter GNN markdown file

## Output
- Standard pipeline mapping: output/22_gui_output/
- constructed_model.md reflects saved edits

## Public API
```python
from gui import (
  run_gui,
  # Markdown helpers
  add_component_to_markdown,
  update_component_states,
  remove_component_from_markdown,
  parse_components_from_markdown,
  # State-space helpers
  parse_state_space_from_markdown,
  add_state_space_entry,
  update_state_space_entry,
  remove_state_space_entry,
)
```

### run_gui
```python
run_gui(
  target_dir: Path,
  output_dir: Path,
  logger: logging.Logger,
  verbose: bool = False,
  headless: bool = False,
  export_filename: str = "constructed_model.md",
  open_browser: bool = True,
) -> bool
```
- Launches the GUI or generates headless artifacts. Returns True on success.

## Architecture
- Files:
  - `processor.py`: Orchestrates GUI launch, headless mode, and integrates with pipeline IO
  - `markdown.py`: Pure helpers for component and state-space markdown manipulation
  - `ui.py`: Gradio UI construction (tabs, inputs, live update wiring)
  - `mcp.py`: Optional MCP registration shim

## Setup via UV
- Step 1 installs GUI extras (`--extra gui`) to include Gradio. To install manually:
```bash
uv sync --extra gui
```

## Notes
- Degrades gracefully if Gradio is not installed (headless artifact generation).
- Designed for modularity: logic isolated in `markdown.py`, UI wiring in `ui.py`, thin orchestration in `processor.py`.
