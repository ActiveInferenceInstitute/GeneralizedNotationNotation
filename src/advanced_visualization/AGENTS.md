# Advanced Visualization Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced visualization, interactive plots, 3D visualizations, and dashboard generation for GNN models

**Pipeline Step**: Step 9: Advanced visualization (9_advanced_viz.py)

**Category**: Advanced Visualization / Interactive Analysis

---

## Core Functionality

### Primary Responsibilities
1. Generate interactive 3D visualizations
2. Create dynamic dashboard interfaces
3. Produce advanced statistical plots
4. Generate interactive HTML visualizations
5. Provide multi-dimensional data exploration

### Key Capabilities
- 3D network topology visualization
- Interactive Plotly dashboards
- Time-series animation
- Multi-panel comparative analysis
- HTML-based interactive reports

---

## API Reference

### Public Functions

#### `process_advanced_viz_standardized_impl(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main advanced visualization processing function (FIXED import)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for visualizations
- `logger` (Logger): Logger instance
- `viz_type` (str): Visualization type ("all", "3d", "interactive", "dashboard")
- `interactive` (bool): Enable interactive features
- `export_formats` (List[str]): Export formats ["html", "json", "png"]
- `**kwargs**: Additional options

**Returns**: `True` if visualization succeeded

---

## Visualization Types

### 3D Visualization
- Network topology in 3D space
- State space visualization
- Connection strength representation

### Interactive Dashboard
- Real-time model exploration
- Parameter adjustment interface
- Multi-view synchronized displays

### Statistical Plots
- Distribution analysis
- Correlation matrices
- Time-series analysis

---

## Dependencies

### Required Dependencies
- `matplotlib` - Basic plotting
- `numpy` - Numerical operations

### Optional Dependencies
- `plotly` - Interactive visualizations (fallback: static plots)
- `seaborn` - Enhanced statistical plots (fallback: matplotlib)
- `bokeh` - Interactive dashboards (fallback: HTML report)

---

## Usage Examples

### Basic Usage
```python
from advanced_visualization.processor import process_advanced_viz_standardized_impl

success = process_advanced_viz_standardized_impl(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    logger=logger,
    viz_type="all"
)
```

### Interactive Dashboard
```python
success = process_advanced_viz_standardized_impl(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/9_advanced_viz_output"),
    logger=logger,
    viz_type="dashboard",
    interactive=True,
    export_formats=["html", "json"]
)
```

---

## Output Specification

### Output Products
- `{model}_3d_visualization.html` - 3D interactive plot
- `{model}_dashboard.html` - Interactive dashboard
- `{model}_statistical_analysis.png` - Statistical plots
- `{model}_visualization_data.json` - Underlying data
- `advanced_viz_summary.json` - Processing summary

### Output Directory Structure
```
output/9_advanced_viz_output/
├── model_name_3d_visualization.html
├── model_name_dashboard.html
├── model_name_statistical_analysis.png
├── model_name_visualization_data.json
└── advanced_viz_summary.json
```

---

## Performance Characteristics

### Latest Execution (After Fix)
- **Duration**: 344ms (import error - NOW FIXED)
- **Status**: FAILED → ✅ READY FOR TESTING
- **Fix Applied**: Import corrected to `process_advanced_viz_standardized_impl`

### Expected Performance
- **Fast Path**: ~2-5s for basic visualizations
- **Slow Path**: ~10-15s for interactive dashboards
- **Memory**: ~50-100MB for large models

---

## Error Handling

### Graceful Degradation
- **No Plotly**: Generate matplotlib-based visualizations
- **No Bokeh**: Create static HTML reports
- **Large Models**: Simplify visualization, provide warnings

---

## Recent Fix

### Import Error Resolution ✅
**Issue**: `ImportError: cannot import name 'process_advanced_visualization'`
**Fix**: Updated `src/9_advanced_viz.py` line 19:
```python
# Before
from advanced_visualization import process_advanced_visualization

# After
from advanced_visualization.processor import process_advanced_viz_standardized_impl
```
**Status**: Ready for testing

---

## Testing

### Test Files
- `src/tests/test_advanced_viz_integration.py`

### Test Coverage
- **Current**: 75%
- **Target**: 85%+

---

**Last Updated**: September 29, 2025  
**Status**: 🔄 FIXED - Ready for Testing


