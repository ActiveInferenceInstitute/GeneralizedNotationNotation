# GUI 3: State Space Design Studio

## Overview

Interactive Gradio-based design studio for building Active Inference models visually. Provides a state-space editor, ontology mapping table, connection topology editor, and one-click GNN export.

## Usage

```bash
# Launch via pipeline
python src/22_gui.py --target-dir input/gnn_files --output-dir output --gui-types gui_3 --verbose

# Launch standalone
python -c "from gui.gui_3 import gui_3; from pathlib import Path; import logging; gui_3(Path('input/gnn_files'), Path('output/22_gui_output'), logging.getLogger())"
```

## Architecture

```
gui_3/
├── __init__.py          # Public API: gui_3(), get_gui_3_info()
├── processor.py         # GUI launch, GNN content loading
└── ui_designer.py       # Gradio UI builder, export/preview handlers
```

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `gui_3()` | `__init__.py` | Main entry point (delegates to `run_gui`) |
| `run_gui()` | `processor.py` | Launch Gradio design studio |
| `build_design_studio()` | `ui_designer.py` | Build the Gradio `Blocks` interface |
| `export_design()` | `ui_designer.py` | Export edited model as GNN markdown |
| `preview_design()` | `ui_designer.py` | Live-preview model structure |
| `get_gui_3_info()` | `__init__.py` | Module metadata |

## Dependencies

- **Optional**: `gradio` — Gradio-based web UI framework
- **Fallback**: Headless mode with log-only output when Gradio unavailable

## References

- [AGENTS.md](AGENTS.md) — Agent documentation
- [Parent GUI Module](../AGENTS.md) — GUI module overview
