# GUI Module (Interactive GNN Constructors)

## Overview
The GUI module provides **three distinct interactive interfaces** for constructing and editing GNN models:

### GUI 1: Form-based Interactive GNN Constructor
- Interactive two-pane editor for GNN models
- Left panel (Controls):
  - Components tab: add/remove components, set type (observation/hidden/action/policy), and manage states
  - State Space tab: live list of state-space entries with name, dimensions, type; add/update/remove entries
- Right panel: synchronized plaintext GNN markdown editor
- Edits are immediately reflected in the markdown
- **Access:** http://localhost:7860

### GUI 2: Visual Matrix Editor
- Visual drag-and-drop matrix editing interface
- Interactive heatmaps and plots for matrix visualization
- Real-time GNN markdown generation from visual edits
- POMDP template-based initialization
- Multi-tab interface for A, B, C, D matrices
- Matrix validation and consistency checking
- **Access:** http://localhost:7861

### GUI 3: State Space Design Studio
- 🎨 **Visual state space designer** with interactive diagrams
- 📚 **Ontology term editor** for Active Inference concepts  
- 🔗 **Connection graph interface** with SVG visualization
- ⚙️ **Parameter tuning controls** (states, observations, actions)
- 💾 **Real-time GNN export and preview** functionality
- 🎯 **Low-dependency HTML/CSS design** approach
- **Access:** http://localhost:7862

## Usage
```bash
# Run all available GUIs (default)
python src/22_gui.py --target-dir input/gnn_files --output-dir output --verbose

# Run specific GUI
python src/22_gui.py --gui-mode gui_1 --target-dir input/gnn_files --output-dir output --verbose
python src/22_gui.py --gui-mode gui_2 --target-dir input/gnn_files --output-dir output --verbose
python src/22_gui.py --gui-mode gui_3 --target-dir input/gnn_files --output-dir output --verbose

# Run multiple specific GUIs
python src/22_gui.py --gui-mode "gui_1,gui_2,gui_3" --target-dir input/gnn_files --output-dir output --verbose
```

- With dependencies available, local web UIs launch (in-browser). Otherwise, headless artifacts are generated.

## Headless Mode
- If GUIs cannot be launched (or headless=True):
  - GUI-specific status JSON files summarize backend availability and export paths
  - Starter GNN markdown files are created for each GUI
  - Visual matrix data exported as JSON (GUI 2)

## Output Structure
- Standard pipeline mapping: output/22_gui_output/
  - `gui_1_output/`: GUI 1 specific outputs (constructed_model_gui_1.md, etc.)
  - `gui_2_output/`: GUI 2 specific outputs (visual_model_gui_2.md, visual_matrices.json, etc.)
  - `gui_3_output/`: GUI 3 specific outputs (designed_model_gui_3.md, design_analysis.json, etc.)
  - `gui_processing_summary.json`: Overall processing summary

## Public API
```python
from gui import (
  # Main processing function
  process_gui,
  
  # Individual GUI runners
  gui_1,
  gui_2,
  gui_3,
  
  # Information functions
  get_available_guis,
  get_gui_1_info,
  get_gui_2_info,
  get_gui_3_info,
  
  # GUI 1 utilities (backward compatibility)
  add_component_to_markdown,
  update_component_states,
  remove_component_from_markdown,
  parse_components_from_markdown,
  parse_state_space_from_markdown,
  add_state_space_entry,
  update_state_space_entry,
  remove_state_space_entry,
)
```

### Individual GUI Functions
```python
gui_1(target_dir: Path, output_dir: Path, logger: Logger, **kwargs) -> Dict[str, Any]
gui_2(target_dir: Path, output_dir: Path, logger: Logger, **kwargs) -> Dict[str, Any]
gui_3(target_dir: Path, output_dir: Path, logger: Logger, **kwargs) -> Dict[str, Any]
```
- Launch specific GUI implementations
- Returns execution results and status information

### process_gui (Main Function)
```python
process_gui(
  target_dir: Path,
  output_dir: Path,
  verbose: bool = False,
  gui_types: List[str] = ['gui_1', 'gui_2'],  # Which GUIs to run
  headless: bool = False,
  open_browser: bool = True,
  **kwargs
) -> bool
```
- Orchestrates execution of multiple GUI implementations
- Returns True if all requested GUIs succeed

## Architecture

### Modular Structure
```
src/gui/
├── __init__.py          # Main module exports and process_gui function
├── README.md           # This file
├── gui_1/              # GUI 1 implementation
│   ├── __init__.py     # GUI 1 exports  
│   ├── processor.py    # GUI 1 main logic
│   ├── markdown.py     # Component and state-space helpers
│   ├── ui.py           # Gradio form-based interface
│   └── mcp.py          # Optional MCP registration
├── gui_2/              # GUI 2 implementation
│   ├── __init__.py     # GUI 2 exports
│   ├── processor.py    # GUI 2 main logic
│   ├── matrix_editor.py # Matrix manipulation and templates
│   ├── ui.py           # Gradio visual interface
│   └── ui_simple.py    # Simplified matrix editor
└── gui_3/              # GUI 3 implementation
    ├── __init__.py     # GUI 3 exports
    ├── processor.py    # GUI 3 main logic
    └── ui_designer.py  # Low-dependency design studio
```

### Adding New GUIs
To add a new GUI (e.g., gui_4):
1. Create `gui_4/` subfolder with implementation
2. Add `gui_4()` function in `gui_4/__init__.py`  
3. Update `gui/__init__.py` to import and expose gui_4
4. Update `22_gui.py` to handle gui_4 in processing logic
5. Add port assignment and URL mapping

**Example completed:** GUI 3 (State Space Design Studio) follows this pattern.

## Setup via UV
- Step 1 installs GUI extras (`--extra gui`) to include Gradio. To install manually:
```bash
uv sync --extra gui
```

## Notes
- Degrades gracefully if Gradio is not installed (headless artifact generation).
- Designed for modularity: logic isolated in `markdown.py`, UI wiring in `ui.py`, thin orchestration in `processor.py`.
