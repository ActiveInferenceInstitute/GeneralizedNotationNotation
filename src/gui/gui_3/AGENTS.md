# GUI 3: State Space Design Studio — Agent Scaffolding

## Module Overview

**Purpose**: Low-dependency visual design experience for constructing Active Inference models via an interactive state-space editor
**Pipeline Step**: Step 22: GUI Processing — gui_3 option (`22_gui.py`)
**Parent Module**: `gui/` (Interactive GNN Constructors)
**Category**: Interactive Visualization / Model Construction
**Status**: ✅ Production Ready
**Version**: 1.6.0
**Last Updated**: 2026-04-15

---

## Core Functionality

### Primary Responsibilities

1. Launch an interactive Gradio-based state-space design studio
2. Parse existing GNN files as starter content for editing
3. Provide visual ontology mapping and connection editing
4. Export designed models as valid GNN markdown files
5. Live-preview of model structure during design

### Key Capabilities

- Gradio-powered GUI (`gr.Blocks`) for visual model construction
- GNN content parsing and pre-population of design fields
- State-space variable editor with dimension and type controls
- Ontology term mapping via editable data tables
- Connection topology editor with visual HTML elements
- One-click GNN export with immediate file output
- Headless fallback when Gradio is unavailable

---

## API Reference

### Public Functions

#### `gui_3(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`

**Description**: Main entry point called by the GUI module aggregator. Delegates to `run_gui()`.

#### `run_gui(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`

**Description**: Launch the State Space Design Studio GUI. Loads starter GNN content from `target_dir`, builds and launches the Gradio interface.

#### `build_design_studio(markdown_text: str, export_path: Path, logger: logging.Logger) -> gr.Blocks`

**Description**: Construct the Gradio `Blocks` UI from parsed GNN content. Returns a launchable Gradio app with state-space, ontology, connection, and export tabs.

#### `get_gui_3_info() -> Dict[str, Any]`

**Description**: Return module metadata (name, version, status, capabilities).

---

## File Structure

```
gui/gui_3/
├── __init__.py          # Public API (gui_3, get_gui_3_info)
├── processor.py         # GUI launch, GNN content loading, analysis
├── ui_designer.py       # Gradio UI construction, export/preview logic
├── AGENTS.md            # This file
└── README.md            # Usage guide
```

## Internal Components

| File | Key Functions | Purpose |
|------|--------------|---------|
| `processor.py` | `run_gui()`, `_load_starter_content()`, `_analyze_gnn_design()` | GUI launch and GNN content loading |
| `ui_designer.py` | `build_design_studio()`, `export_design()`, `preview_design()` | Gradio UI construction and event handlers |

---

## Dependencies

- **Required**: `pathlib`, `json`
- **Optional**: `gradio` (recovery: headless mode with log-only output)

---

## Integration Points

- **Orchestrated By**: `22_gui.py` → `gui/__init__.py` → `gui_3()`
- **Upstream**: Consumes GNN files from `input/gnn_files/`
- **Downstream**: Exports GNN `.md` files to `output/22_gui_output/`

---

## Documentation

- **[README](README.md)**: Usage guide
- **[AGENTS](AGENTS.md)**: This file
