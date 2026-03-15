# Step 22: GUI — Interactive GNN Constructor

## Overview

Provides multiple interactive GUI implementations built with Gradio, plus headless artifact generation mode for pipeline integration. Includes form-based constructors, visual matrix editors, state space design studios, and Oxdraw diagram-as-code.

## Usage

```bash
# Headless mode (pipeline default)
python src/22_gui.py --target-dir input/gnn_files --output-dir output --headless

# Interactive mode (launch GUI servers)
python src/22_gui.py --target-dir input/gnn_files --output-dir output --interactive

# Specific GUI types
python src/22_gui.py --gui-types "gui_1,oxdraw" --interactive --open-browser
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/22_gui.py` (106 lines) |
| Module | `src/gui/` |
| Processor | `src/gui/processor.py` |
| Module function | `process_gui()` |

## Available GUI Types

| Type | Description | Port |
|------|-------------|:----:|
| `gui_1` | Form-based Interactive GNN Constructor | 7860 |
| `gui_2` | Visual Matrix Editor | 7861 |
| `gui_3` | State Space Design Studio | 7862 |
| `oxdraw` | Visual diagram-as-code (Mermaid) | 5151 |

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--headless` | flag | `False` | Generate artifacts only, no GUI servers |
| `--interactive` | flag | `False` | Launch interactive GUI servers |
| `--gui-types` | `str` | `gui_1,gui_2` | Comma-separated GUI types to run |
| `--open-browser` | flag | `False` | Auto-open browser for interactive GUIs |

## Optional Dependencies

Install with: `uv pip install -e .[gui]` (requires gradio, plotly, numpy, pandas)

For Oxdraw: `cargo install oxdraw`

## Output

- **Directory**: `output/22_gui_output/`
- `gui_processing_summary.json`, constructed GNN models, GUI metadata

## Source

- **Script**: [src/22_gui.py](../../src/22_gui.py)
