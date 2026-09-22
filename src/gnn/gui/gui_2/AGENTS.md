# GUI 2: Visual Matrix Editor - Agent Scaffolding

## Module Overview

**Purpose**: Provides an advanced, interactive visual interface for direct manipulation of Active Inference vectors, tensors, and connection mapping.

**Pipeline Step**: Step 22: GUI Processing (22_gui.py)

**Category**: Interactive Visualization / Model Construction

**Status**: Production Ready

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

---

## Core Functionality

### Primary Responsibilities

1. **Visual Matrix Representation**: Renders parameter spaces (like the A, B matrices) as interactive Plotly heatmaps and grids.
2. **Tabbed Matrix Editing**: Organizes A, B, C, and D parameter editors into tabs with +/- dimension controls, heatmap previews, and value-preserving resize.
3. **Template Initialization**: Bootstraps blank interfaces with structurally-sound Active Inference Discrete POMDP priors (e.g. valid probability distributions).

### Key Capabilities

- Interactive Plotly heatmaps for matrix values and bar charts for vectors.
- Integrated validation via `validate_visual_matrix_dimensions`: finite values, declared shapes, and cross-matrix dimension consistency (A rows vs C, A columns vs D, B rows/columns vs D).
- Real-time regeneration of the GNN markdown from edited tables via `update_gnn_from_matrix`.

---

## API Reference

### Public Functions

#### `gui_2(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`

**Description**: The primary entry point invoked by the pipeline's Step 22 GUI processing; routes parameters (headless, export_filename, open_browser) into `run_gui` and returns a result dict with status, backend, and output file path.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to load (prefers POMDP templates).
- `output_dir` (Path): Output directory for results.
- `logger` (logging.Logger): Unified pipeline logger.
- `**kwargs`: Optional settings — `headless`, `export_filename` (default `visual_model_gui2.md`), `open_browser`, `verbose`.

---

## Dependencies

- **gradio**: Web server UI framework (required for interactive mode).
- **plotly**: Interactive heatmaps and vector charts (optional; matrix/vector data grids remain fully usable without it).
- **numpy**: Matrix array handling for plot construction.

---

## Integration Points

- Plugs into `gui.__init__.py` alongside `gui_1` and `gui_3`.
- Exports standard format compatible directly with `Step 5 (Type Checker)` and `Step 12 (Execute)`.
