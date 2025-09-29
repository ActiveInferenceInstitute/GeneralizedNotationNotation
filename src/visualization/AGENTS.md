# Visualization Module - Agent Scaffolding

## Module Overview

**Purpose**: Generate matrix visualizations, network graphs, and combined analysis plots from GNN models

**Pipeline Step**: Step 8: Visualization (8_visualization.py)

**Category**: Visualization / Analysis

---

## Core Functionality

### Primary Responsibilities
1. Generate matrix heatmaps for model parameters
2. Create network graphs of variable connections
3. Produce combined visualization analyses
4. Safe matplotlib DPI handling
5. POMDP-aware visualizations

### Key Capabilities
- Matrix visualization with heatmaps
- Network topology visualization
- Combined multi-panel visualizations
- Safe plot saving with DPI sanitization
- Graceful fallback when dependencies unavailable

---

## API Reference

### Public Functions

#### `process_visualization_main(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main visualization processing function

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for visualizations
- `logger` (Logger): Logger instance
- `**kwargs`: Additional options

**Returns**: `True` if visualization succeeded

#### `_save_plot_safely(fig, output_path, dpi=300, logger=None)`
**Description**: Safely save matplotlib figure with DPI sanitization

**Parameters**:
- `fig`: Matplotlib figure
- `output_path` (Path): Output file path
- `dpi` (int/float): Target DPI (sanitized automatically)
- `logger`: Optional logger

---

## Dependencies

### Required Dependencies
- `matplotlib` - Plotting library
- `networkx` - Network graph visualization
- `numpy` - Numerical operations

### Optional Dependencies
- `seaborn` - Enhanced visualizations (fallback: basic matplotlib)

---

## Usage Examples

### Basic Usage
```python
from visualization import process_visualization_main

success = process_visualization_main(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/8_visualization_output"),
    logger=logger
)
```

---

## Output Specification

### Output Products
- `*_matrix_analysis.png` - Matrix heatmap visualizations
- `*_network_graph.png` - Network topology visualizations
- `*_combined_analysis.png` - Combined multi-panel plots
- `visualization_summary.json` - Processing summary

### Output Directory Structure
```
output/8_visualization_output/
├── model_name_matrix_analysis.png
├── model_name_network_graph.png
├── model_name_combined_analysis.png
└── visualization_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 50.5s (includes POMDP analysis)
- **Memory**: Peak 28.9 MB, Final 14.25 MB
- **Status**: SUCCESS
- **Files Processed**: 1
- **Images Generated**: 5

### DPI Handling
- **Default DPI**: 300
- **Fallback DPI**: 96 (if matplotlib error)
- **Safe Range**: 1-600 DPI

---

## Error Handling

### Safe DPI Sanitization
Handles matplotlib DPI errors by:
1. Attempting save with requested DPI
2. On error, sanitize DPI to safe range (1-600)
3. Retry with sanitized DPI
4. Final fallback to DPI=96

---

## Testing

### Test Files
- `src/tests/test_visualization_integration.py`

### Test Coverage
- **Current**: 84%
- **Target**: 90%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready


