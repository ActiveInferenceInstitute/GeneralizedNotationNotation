# GUI 1: Visual Matrix Editor - Agent Scaffolding

## Module Overview

**Purpose**: Provides an advanced, click-and-drag visual interface specifically tuned for direct manipulation of Active Inference vectors, tensors, and connection mapping.

**Pipeline Step**: Step 22: GUI Processing (22_gui.py)

**Category**: Interactive Visualization / Model Construction

**Status**: ✅ Production Ready

**Version**: 1.3.0

---

## Core Functionality

### Primary Responsibilities

1. **Visual Matrix Representation**: Renders parameter spaces (like the A, B matrices) as interactive Plotly heatmaps and grids.
2. **Drag-and-Drop Editing**: Facilitates hierarchical restructuring by dragging visual state items into valid domains.
3. **Template Initialization**: Bootstraps blank interfaces with structurally-sound Active Inference Discrete POMDP priors (e.g. valid probability distributions).

### Key Capabilities

- Interactive heatmaps displaying precision parameters and parameter biases.
- Integrated validations to ensure matrices sum to 1.0 where required by the ontology.
- Real-time serialization directly bypassing intermediate strings.

---

## API Reference

### Public Functions

#### `gui_2(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`

**Description**: The primary entry-point mapping variables from `22_gui.py` into the physical web server execution environment.

**Parameters**:
- `target_dir` (Path): Path representing where template models reside.
- `output_dir` (Path): Target export path.
- `logger` (logging.Logger): Unified pipeline logger.

---

## Dependencies

- **gradio**: Web socket orchestration.
- **plotly**: Visual heatmapping and advanced graph displays.
- **numpy**: Matrix representation tracking before GNN dumping.

---

## Integration Points

- Plugs into `gui.__init__.py` alongside `gui_1` and `gui_3`.
- Exports standard format compatible directly with `Step 5 (Type Checker)` and `Step 12 (Execute)`.
