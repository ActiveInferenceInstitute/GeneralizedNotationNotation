# GUI 1: Interactive GNN Constructor - Agent Scaffolding

## Module Overview

**Purpose**: Provides the primary form-based visual interface for constructing and editing Active Inference generative models. It features a Gradio-based application allowing real-time editing of GNN components, state spaces, and immediate markdown synchronization.

**Pipeline Step**: Step 22: GUI Processing (22_gui.py)

**Category**: Interactive Visualization / Model Construction

**Status**: ✅ Production Ready

**Version**: 1.0.0

---

## Core Functionality

### Primary Responsibilities

1. **Interactive Model Editing**: Renders generalized model components (matrices, states, observations, connections) into editable form fields.
2. **State Space Validation**: Dynamically checks dimension bounds and state space configurations.
3. **Live Markdown Synchronization**: Two-pane interface synchronizing web changes directly to the target `.md` file representation.

### Key Capabilities

- Two-pane Gradio web interface (Controls + Markdown).
- Form-based component management.
- Live validation mapping to Active Inference terminology.
- Integration as a headless backend or interactive server.

### Agent Capabilities

This module provides specialized agent capabilities for visual construction:

#### 🎯 Form Interface Agent
- **Core Function**: Maps structured dict strings into visual schemas.
- **Input Processing**: Processes `target_dir` GNN files.
- **Output Generation**: Emits rewritten Markdown content and runs internal parser checks.

---

## API Reference

### Public Functions

#### `gui_1(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`

**Description**: Main orchestrator to run GUI 1. Handles parameter extraction for headless vs. browser mode.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to load.
- `output_dir` (Path): Output directory for the saved models.
- `logger` (logging.Logger): Standard pipeline logger.
- `**kwargs`: Supported kwargs include `headless` (bool), `export_filename` (str), and `open_browser` (bool).

**Returns**: A dictionary containing `gui_type`, `success`, `output_file`, and execution details.

#### `get_gui_1_info() -> Dict[str, Any]`

**Description**: Exposes metadata capabilities (e.g. features, requirements, layout type) to the aggregator.

---

## Dependencies

### Required Dependencies
- `gradio` - Core web framework for the interactive application.
- `pathlib` - Filesystem resolution.

### Internal Dependencies
- `gui.gui_1.markdown` - Bi-directional text synchronization algorithms.
- `gui.gui_1.processor` - Controller logic coordinating Gradio and standard logic.

---

## Configuration

### Default Settings

```python
# Headless defaults
headless = False
export_filename = 'constructed_model_gui1.md'
open_browser = True
```

---

## Error Handling

### Recovery Strategies
- **Parse Failures**: If standard GNN parsing fails mid-edit, the text window relies on a "last known good state" buffer, presenting errors directly in the Gradio console output pane.
- **Dependency Missing**: Raises standard Python `ImportError` gracefully captured by `22_gui.py` fallback logs.

---

## Development Guidelines

### Adding New Component Forms
1. Update `markdown.py` to support inserting the component text safely.
2. Register the Gradio inputs in `ui.py`.
3. Add to the `__all__` exported registry in `__init__.py`.

**Last Updated**: 2026-04-10
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
