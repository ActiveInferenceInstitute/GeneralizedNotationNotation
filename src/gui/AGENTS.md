# GUI Module - Agent Scaffolding

## Module Overview

**Purpose**: Interactive graphical user interfaces for visual GNN model construction and editing with multiple specialized implementations

**Pipeline Step**: Step 22: GUI (Interactive GNN Constructor) (22_gui.py)

**Category**: Interactive Visualization / Model Construction

---

## Core Functionality

### Primary Responsibilities
1. Provide multiple GUI implementations for GNN model construction
2. Enable visual editing of model components and state spaces
3. Generate interactive visualizations for model exploration
4. Support real-time model validation and feedback
5. Export constructed models to GNN format

### Key Capabilities
- Form-based interactive GNN constructor (GUI 1)
- Visual matrix editor with drag-and-drop (GUI 2)
- State space design studio (GUI 3)
- Real-time model validation and error checking
- Interactive matrix heatmaps and network visualizations
- Component management and state space editing

---

## API Reference

### Public Functions

#### `process_gui(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main GUI processing function that runs all available GUI implementations

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for GUI results
- `logger` (Logger): Logger instance for progress reporting
- `gui_mode` (str): GUI mode ("all", "gui_1", "gui_2", "gui_3")
- `interactive_mode` (bool): Run GUIs in interactive mode
- `headless` (bool): Run in headless mode (artifact generation only)
- `**kwargs`: Additional GUI-specific options

**Returns**: `True` if GUI processing succeeded

#### `gui_1(target_dir, output_dir, logger, **kwargs) -> Dict[str, Any]`
**Description**: Form-based Interactive GNN Constructor (GUI 1)

**Parameters**:
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory for GUI 1
- `logger` (Logger): Logger instance
- `verbose` (bool): Enable verbose logging
- `headless` (bool): Run in headless mode
- `export_filename` (str): Output filename for constructed model
- `open_browser` (bool): Open browser for interactive mode

**Returns**: Dictionary with GUI 1 execution results

#### `gui_2(target_dir, output_dir, logger, **kwargs) -> Dict[str, Any]`
**Description**: Visual Matrix Editor with drag-and-drop interface (GUI 2)

**Parameters**:
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory for GUI 2
- `logger` (Logger): Logger instance
- `verbose` (bool): Enable verbose logging
- `headless` (bool): Run in headless mode
- `export_filename` (str): Output filename for visual model
- `open_browser` (bool): Open browser for interactive mode

**Returns**: Dictionary with GUI 2 execution results

#### `get_available_guis() -> Dict[str, Dict]`
**Description**: Get information about all available GUI implementations

**Returns**: Dictionary with GUI information including names, descriptions, and ports

---

## GUI Implementations

### GUI 1: Form-based Constructor
**Port**: `:7860`
**Focus**: Step-by-step model building with component management
**Features**:
- Interactive two-pane editor (components + markdown)
- Component management (observation/hidden/action/policy variables)
- State space entry management with live validation
- Synchronized plaintext GNN markdown editor
- Real-time model validation and error feedback

### GUI 2: Visual Matrix Editor
**Port**: `:7861`
**Focus**: Real-time matrix heatmaps and interactive editing
**Features**:
- Interactive DataFrame editing with +/- dimension controls
- Vector bar chart displays for C & D vectors
- Live matrix statistics (min, max, mean, sum)
- Auto-update functionality and matrix validation
- Drag-and-drop matrix manipulation

### GUI 3: State Space Design Studio
**Port**: `:7862`
**Focus**: Visual state space architecture design
**Features**:
- Visual state space architecture designer with SVG diagrams
- Ontology term editor for Active Inference concept mapping
- Interactive connection graph interface (D>s, s-A, A-o format)
- Parameter tuning controls (states, observations, actions, horizons)
- Real-time GNN export and preview with low-dependency approach

---

## Dependencies

### Required Dependencies
- `gradio` - Web-based GUI framework
- `pandas` - Data manipulation for matrix editing
- `numpy` - Numerical operations for visualizations

### Optional Dependencies
- `plotly` - Interactive visualizations (fallback: matplotlib)
- `matplotlib` - Static plotting (fallback: basic HTML)
- `networkx` - Network graph visualization

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `GUI_DEFAULT_PORT` - Default port for GUI servers (7860)
- `GUI_HEADLESS_MODE` - Run GUIs in headless mode by default
- `GUI_BROWSER_AUTO_OPEN` - Automatically open browser for interactive GUIs

### Configuration Files
- `gui_config.yaml` - GUI-specific settings and themes

### Default Settings
```python
DEFAULT_GUI_SETTINGS = {
    'gui_1': {
        'port': 7860,
        'theme': 'default',
        'auto_save': True,
        'validation_enabled': True
    },
    'gui_2': {
        'port': 7861,
        'heatmap_colormap': 'viridis',
        'matrix_precision': 3,
        'auto_update': True
    },
    'gui_3': {
        'port': 7862,
        'diagram_style': 'modern',
        'ontology_integration': True,
        'export_format': 'markdown'
    }
}
```

---

## Usage Examples

### Basic Usage (All GUIs)
```python
from gui import process_gui

success = process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    gui_mode="all",
    interactive_mode=True
)
```

### Specific GUI (GUI 1)
```python
from gui.gui_1 import gui_1

result = gui_1(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/gui_1_output"),
    logger=logger,
    headless=False,
    export_filename="constructed_model.md",
    open_browser=True
)
```

### GUI Information Query
```python
from gui import get_available_guis

guis = get_available_guis()
for gui_name, info in guis.items():
    print(f"{gui_name}: {info['description']} (Port: {info['port']})")
```

---

## Output Specification

### Output Products
- `{gui_type}_output/constructed_model_{gui_type}.md` - Generated GNN model
- `{gui_type}_output/gui_status.json` - GUI execution status
- `{gui_type}_output/visual_matrices.json` - Matrix data (GUI 2)
- `{gui_type}_output/design_analysis.json` - Design metadata (GUI 3)
- `gui_processing_summary.json` - Overall GUI processing summary

### Output Directory Structure
```
output/22_gui_output/
├── gui_1_output/
│   ├── constructed_model_gui_1.md
│   └── gui_status.json
├── gui_2_output/
│   ├── visual_model_gui_2.md
│   └── visual_matrices.json
├── gui_3_output/
│   ├── designed_model_gui_3.md
│   └── design_analysis.json
└── gui_processing_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-2 seconds in headless mode, ~10-30 seconds for interactive GUI startup
- **Memory**: <10MB for headless mode, ~150-250MB for interactive GUIs
- **Status**: ✅ Ready (Fixed - headless mode default for pipeline)

### Expected Performance
- **Interactive Mode**: ~10-30s startup time per GUI
- **Headless Mode**: ~2-5s for artifact generation
- **Memory**: ~50-100MB for headless, ~150-250MB for interactive

---

## Error Handling

### Graceful Degradation
- **No gradio**: Fallback to HTML-based interfaces
- **No plotly**: Use matplotlib for visualizations
- **Browser unavailable**: Generate static artifacts only

### Error Categories
1. **Dependency Errors**: Missing GUI framework dependencies
2. **Port Conflicts**: GUI server port already in use
3. **Browser Errors**: Unable to launch interactive interface
4. **File System Errors**: Unable to write output files

---

## Integration Points

### Orchestrated By
- **Script**: `22_gui.py` (Step 22)
- **Function**: `process_gui()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_gui_integration.py` - GUI integration tests
- `main.py` - Pipeline orchestration

### Data Flow
```
GNN Files → GUI Construction → Visual Editing → Model Validation → GNN Export → Pipeline Integration
```

---

## Testing

### Test Files
- `src/tests/test_gui_integration.py` - Integration tests
- `src/tests/test_gui_1_unit.py` - GUI 1 unit tests
- `src/tests/test_gui_2_unit.py` - GUI 2 unit tests
- `src/tests/test_gui_3_unit.py` - GUI 3 unit tests

### Test Coverage
- **Current**: 68%
- **Target**: 85%+

### Key Test Scenarios
1. GUI startup and shutdown in headless mode
2. Component creation and editing workflows
3. Matrix manipulation and validation
4. Model export and format validation
5. Error handling with missing dependencies

---

## MCP Integration

### Tools Registered
- `gui_construct` - Launch GNN model constructor
- `gui_edit` - Edit existing GNN models visually
- `gui_validate` - Validate models through GUI interface

### Tool Endpoints
```python
@mcp_tool("gui_construct")
def construct_model_gui(model_type="pymdp", interactive=True):
    """Launch GUI for constructing new GNN model"""
    # Implementation
```

---

## Recent Improvements (September 30, 2025)

### GUI Timeout Fix
- **Problem**: GUI module was timing out in pipeline execution (600s timeout)
- **Root Cause**: GUIs were launching interactive servers with blocking threads even in pipeline mode
- **Solution**: 
  - Implemented proper headless mode that defaults to True when run from pipeline
  - Headless mode generates artifacts only (no server launch, no blocking)
  - Added `--interactive` flag to explicitly enable GUI servers when needed
  - Separated GUI kwargs to avoid parameter conflicts with logger
- **Result**: GUI step now completes in ~1.28s (vs 600s timeout), 99.8% improvement

### Performance Improvements
- Headless mode execution: < 2 seconds (down from 600s timeout)
- Memory usage in headless mode: < 10MB (down from 150-250MB)
- Pipeline integration: No blocking, no timeouts
- Interactive mode still fully functional when explicitly requested

---

**Last Updated**: September 30, 2025
**Status**: ✅ Ready (Timeout Issue Fixed - Headless Mode Default)
