# GUI Module - Agent Scaffolding

## Module Overview

**Purpose**: Interactive graphical user interfaces for visual GNN model construction and editing with multiple specialized implementations

**Pipeline Step**: Step 22: GUI (Interactive GNN Constructor) (22_gui.py)

**Category**: Interactive Visualization / Model Construction

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2026-01-21

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

#### `process_gui(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main GUI processing function called by orchestrator (22_gui.py). Runs all available GUI implementations.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for GUI results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `gui_types` (str, optional): Comma-separated list of GUI types ("gui_1", "gui_2", "gui_3", "oxdraw") (default: "gui_1,gui_2")
- `interactive` (bool, optional): Run GUIs in interactive mode (default: False)
- `headless` (bool, optional): Run in headless mode - artifact generation only (default: True if not interactive)
- `open_browser` (bool, optional): Automatically open browser for interactive GUIs (default: False)
- `**kwargs`: Additional GUI-specific options

**Returns**: `bool` - True if GUI processing succeeded, False otherwise

**Example**:
```python
from gui import process_gui
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
# Headless mode (pipeline default)
success = process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    verbose=True,
    headless=True
)

# Interactive mode
success = process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    interactive=True,
    gui_types="gui_1,oxdraw",
    open_browser=True
)
```

#### `gui_1(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`
**Description**: Form-based Interactive GNN Constructor (GUI 1). Two-pane editor with component management.

**Parameters**:
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory for GUI 1
- `logger` (logging.Logger): Logger instance
- `verbose` (bool, optional): Enable verbose logging (default: False)
- `headless` (bool, optional): Run in headless mode (default: True)
- `export_filename` (str, optional): Output filename for constructed model (default: "constructed_model_gui1.md")
- `open_browser` (bool, optional): Open browser for interactive mode (default: False)
- `port` (int, optional): Port for web server (default: 7860)

**Returns**: `Dict[str, Any]` - GUI 1 execution results with:
- `success` (bool): Whether GUI execution succeeded
- `output_file` (Path): Path to generated GNN model
- `status` (str): Execution status
- `url` (Optional[str]): Web UI URL if interactive

#### `gui_2(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`
**Description**: Visual Matrix Editor with drag-and-drop interface (GUI 2). Matrix heatmaps and POMDP template support.

**Parameters**:
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory for GUI 2
- `logger` (logging.Logger): Logger instance
- `verbose` (bool): Enable verbose logging
- `headless` (bool): Run in headless mode
- `export_filename` (str): Output filename for visual model
- `open_browser` (bool): Open browser for interactive mode

**Returns**: Dictionary with GUI 2 execution results

#### `get_available_guis() -> Dict[str, Dict]`
**Description**: Get information about all available GUI implementations

**Returns**: Dictionary with GUI information including names, descriptions, and ports

#### `generate_html_navigation(pipeline_output_dir, output_dir, logger) -> bool`
**Description**: Generate HTML navigation page that links to all pipeline output types

**Parameters**:
- `pipeline_output_dir` (Path): Directory containing all pipeline outputs (typically output/)
- `output_dir` (Path): GUI output directory where navigation.html will be created
- `logger` (Logger): Logger instance for progress reporting

**Returns**: `True` if navigation page generated successfully, `False` otherwise

**Features**:
- Scans all 24 pipeline output directories
- Discovers files by type (JSON, MD, PNG, SVG, HTML, etc.)
- Generates organized HTML navigation with file metadata
- Provides links to all pipeline artifacts
- Includes summary statistics and integration with comprehensive reports

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

#### GUI-Specific Outputs
- `constructed_model_gui1.md` - Generated GNN model from GUI 1 (Form-based Constructor)
- `visual_model_gui2.md` - Generated GNN model from GUI 2 (Visual Matrix Editor)
- `visual_matrices.json` - Matrix data and visualizations from GUI 2
- `gui_status.json` - GUI execution status and backend information
- `gui_processing_summary.json` - Overall GUI processing summary with results from all GUIs

#### Navigation and Discovery
- `navigation.html` - **Comprehensive HTML navigation page** that provides:
  - Links to all pipeline output files across all 24 steps
  - Organized by pipeline step with file counts and metadata
  - File type filtering and search capabilities
  - Direct links to visualizations, reports, execution results, and all artifacts
  - Summary statistics of pipeline outputs
  - Links to comprehensive reports

### Output Directory Structure
```
output/22_gui_output/
├── constructed_model_gui1.md          # GUI 1 output: Form-based constructed model
├── visual_model_gui2.md               # GUI 2 output: Visual matrix editor model
├── visual_matrices.json                # GUI 2 output: Matrix data and visualizations
├── gui_status.json                     # GUI execution status and backend info
├── gui_processing_summary.json         # Overall processing summary
└── navigation.html                     # HTML navigation to all pipeline outputs
```

### Navigation.html Features

The `navigation.html` file provides comprehensive navigation to all pipeline outputs:

1. **Pipeline Overview**: Summary statistics showing total pipeline steps and files
2. **Output Sections**: Organized by all 24 pipeline steps:
   - Template (0_template_output)
   - Setup (1_setup_output)
   - Tests (2_tests_output)
   - GNN Processing (3_gnn_output)
   - Model Registry (4_model_registry_output)
   - Type Checker (5_type_checker_output)
   - Validation (6_validation_output)
   - Export (7_export_output)
   - Visualization (8_visualization_output)
   - Advanced Visualization (9_advanced_viz_output)
   - Ontology (10_ontology_output)
   - Render (11_render_output)
   - Execute (12_execute_output)
   - LLM (13_llm_output)
   - ML Integration (14_ml_integration_output)
   - Audio (15_audio_output)
   - Analysis (16_analysis_output)
   - Integration (17_integration_output)
   - Security (18_security_output)
   - Research (19_research_output)
   - Website (20_website_output)
   - MCP (21_mcp_output)
   - GUI (22_gui_output)
   - Report (23_report_output)

3. **File Information**: For each file, displays:
   - File name with clickable link
   - File type/extension
   - File size in MB
   - Relative path for navigation

4. **Integration**: Links to comprehensive report at `23_report_output/comprehensive_analysis_report.html`

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

## Recent Enhancements (January 5, 2026)

### HTML Navigation Generation
- **Added**: `generate_html_navigation()` function to create comprehensive navigation page
- **Features**:
  - Scans all 24 pipeline output directories automatically
  - Discovers and catalogs all output files by type
  - Generates organized HTML with file metadata (type, size, path)
  - Provides direct links to all pipeline artifacts
  - Includes summary statistics and integration with comprehensive reports
- **Output**: `navigation.html` file in GUI output directory
- **Integration**: Automatically generated during `process_gui()` execution
- **Access**: Open `output/22_gui_output/navigation.html` in web browser for full pipeline navigation

### Output File Discovery
- **Enhanced**: All output files are now properly discovered and documented
- **Files Generated**:
  - `constructed_model_gui1.md` - GUI 1 form-based model construction
  - `visual_model_gui2.md` - GUI 2 visual matrix editor model
  - `visual_matrices.json` - GUI 2 matrix data and visualizations
  - `gui_status.json` - GUI execution status and backend information
  - `gui_processing_summary.json` - Overall processing summary
  - `navigation.html` - Comprehensive navigation to all pipeline outputs

---

## MCP Integration

### Tools Registered
- `gui.construct_model` - Construct GNN model using GUI
- `gui.edit_component` - Edit model component
- `gui.visualize_model` - Visualize model structure

### Tool Endpoints
```python
@mcp_tool("gui.construct_model")
def construct_model_tool(components: List[Dict]) -> Dict[str, Any]:
    """Construct GNN model from components"""
    # Implementation
```

### MCP File Location
- `src/gui/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: GUI fails to launch
**Symptom**: GUI server doesn't start or browser doesn't open  
**Cause**: Port already in use or dependencies missing  
**Solution**: 
- Check if port is already in use: `lsof -i :7860`
- Use different port: `--port 7861`
- Verify Gradio/Streamlit dependencies installed
- Use `--headless` mode if GUI not needed

#### Issue 2: Model export fails
**Symptom**: GUI completes but model file not generated  
**Cause**: Export function errors or file permissions  
**Solution**:
- Check output directory permissions
- Verify export filename is valid
- Review export function logs
- Ensure GNN format validation passes

---

## Version History

### Current Version: 1.0.0

**Features**:
- Multiple GUI implementations (GUI 1, GUI 2, GUI 3, oxdraw)
- Interactive model construction
- Real-time validation
- Model export to GNN format

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced visual editing
- **Future**: Collaborative editing

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [GUI Guide](../../doc/gui_oxdraw/)

### External Resources
- [Gradio Documentation](https://gradio.app/)
- [Streamlit Documentation](https://streamlit.io/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
