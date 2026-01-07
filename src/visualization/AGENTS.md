# Visualization Module - Agent Scaffolding

## Module Overview

**Purpose**: Graph and matrix visualization generation for GNN models

**Pipeline Step**: Step 8: Visualization (8_visualization.py)

**Category**: Visualization / Graph Analysis

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

---

## Core Functionality

### Primary Responsibilities
1. Generate graph visualizations from GNN models
2. Create matrix heatmaps and plots
3. Visualize model structure and connections
4. Generate network topology diagrams
5. Provide visualization data for advanced analysis

### Key Capabilities
- Network graph generation and layout
- Matrix visualization and heatmap creation
- Interactive visualization support
- Multiple output formats (PNG, SVG, HTML)
- Model structure visualization

---

## API Reference

### Public Functions

#### `process_visualization_main(target_dir, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Main visualization processing function called by orchestrator (8_visualization.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for visualizations
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional visualization options

**Returns**: `True` if visualization succeeded

**Example**:
```python
from visualization import process_visualization_main

success = process_visualization_main(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/8_visualization_output"),
    verbose=True
)
```

#### `generate_graph_visualization(graph_data) -> List[str]`
**Description**: Generate graph visualization from graph data

**Parameters**:
- `graph_data`: Graph data dictionary

**Returns**: List of generated visualization file paths

#### `generate_matrix_visualization(matrix_data) -> List[str]`
**Description**: Generate matrix visualization from matrix data

**Parameters**:
- `matrix_data`: Matrix data dictionary

**Returns**: List of generated visualization file paths

#### `create_network_diagram(graph_data) -> Dict[str, Any]`
**Description**: Create network diagram visualization

**Parameters**:
- `graph_data`: Graph data dictionary

**Returns**: Dictionary with visualization results

---

## Dependencies

### Required Dependencies
- `matplotlib` - Plotting and visualization
- `networkx` - Network graph algorithms
- `numpy` - Numerical computations

### Optional Dependencies
- `plotly` - Interactive visualizations
- `graphviz` - Graph layout and rendering

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Visualization Settings
```python
VISUALIZATION_CONFIG = {
    'output_format': 'png',
    'dpi': 300,
    'figsize': (10, 8),
    'colormap': 'viridis',
    'layout_algorithm': 'spring'
}
```

### Graph Settings
```python
GRAPH_CONFIG = {
    'node_size': 100,
    'edge_width': 1,
    'node_color': 'lightblue',
    'edge_color': 'gray',
    'layout': 'force_directed'
}
```

---

## Usage Examples

### Basic Visualization
```python
from visualization import process_visualization_main

success = process_visualization_main(
    target_dir="input/gnn_files",
    output_dir="output/8_visualization_output"
)
```

### Graph Visualization
```python
from visualization import generate_graph_visualization

files = generate_graph_visualization(graph_data)
for file_path in files:
    print(f"Generated: {file_path}")
```

### Matrix Visualization
```python
from visualization import generate_matrix_visualization

files = generate_matrix_visualization(matrix_data)
for file_path in files:
    print(f"Generated: {file_path}")
```

---

## Output Specification

### Output Products
- `*_network.png` - Network graph visualizations
- `*_matrix.png` - Matrix heatmap visualizations
- `*_structure.png` - Model structure visualizations
- `visualization_summary.json` - Visualization summary

### Output Directory Structure
```
output/8_visualization_output/
├── model_name_network.png
├── model_name_matrix.png
├── model_name_structure.png
├── visualization_summary.json
└── detailed_analysis/
    ├── graph_data.json
    └── matrix_data.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 seconds per model
- **Memory**: ~50-150MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Graph Generation**: 1-3 seconds
- **Matrix Visualization**: 1-2 seconds
- **Structure Analysis**: 2-4 seconds
- **Combined Visualization**: 3-6 seconds

---

## Error Handling

### Visualization Errors
1. **Graph Layout**: Graph layout algorithm failures
2. **Matrix Size**: Matrix too large for visualization
3. **File I/O**: Visualization file writing failures
4. **Dependency**: Missing visualization dependencies

### Recovery Strategies
- **Layout Fallback**: Use simpler layout algorithms
- **Matrix Sampling**: Sample large matrices
- **Format Fallback**: Try alternative output formats
- **Dependency Skip**: Skip advanced visualizations

---

## Integration Points

### Orchestrated By
- **Script**: `8_visualization.py` (Step 8)
- **Function**: `process_visualization_main()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `advanced_visualization` - Advanced visualization module
- `tests.test_visualization_*` - Visualization tests

### Data Flow
```
GNN Files → Graph Extraction → Layout Calculation → Visualization Generation → Output Files
```

---

## Testing

### Test Files
- `src/tests/test_visualization_integration.py` - Integration tests
- `src/tests/test_visualization_matrices.py` - Matrix visualization tests
- `src/tests/test_visualization_comprehensive.py` - Comprehensive real-data tests ✨ NEW

### Test Coverage
- **Current**: 84%
- **Target**: 90%+

### Key Test Scenarios
1. Graph visualization with various layouts
2. Matrix heatmap generation
3. Model structure visualization
4. Error handling and recovery
5. Matplotlib backend configuration
6. Headless environment support
7. Progress tracking validation

---

## MCP Integration

### Tools Registered
- `visualization.generate_graph` - Generate graph visualization
- `visualization.generate_matrix` - Generate matrix visualization
- `visualization.create_network` - Create network diagram
- `visualization.analyze_structure` - Analyze model structure

### Tool Endpoints
```python
@mcp_tool("visualization.generate_graph")
def generate_graph_tool(graph_data):
    """Generate graph visualization"""
    # Implementation
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Matplotlib Backend Warnings
**Symptom**: Warnings about matplotlib backend or "no DISPLAY" errors

**Solution**:
- ✅ **Automatic Fix**: The module now automatically detects headless environments and configures the `Agg` backend
- Environment variable: Set `MPLBACKEND=Agg` before running
- Manual fix: Add to your script:
  ```python
  import matplotlib
  matplotlib.use('Agg')
  ```

**Prevention**: Run in environments with display support or ensure `Agg` backend is used

#### 2. Missing Visualization Dependencies
**Symptom**: ImportError for matplotlib, networkx, or numpy

**Solution**:
```bash
pip install matplotlib>=3.5.0 networkx>=2.8.0 numpy>=1.21.0
```

**Alternative**: Install all visualization dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Large Model Visualization Failures
**Symptom**: Visualization fails or hangs with large models (>100 nodes)

**Solution**:
- ✅ **Automatic**: Module samples large models automatically
- Manual override: Set sampling parameters in config
- Alternative: Visualize model subsets

**Prevention**: Use `--sample-large-models` flag when processing

#### 4. Memory Issues During Visualization
**Symptom**: Out of memory errors or system slowdown

**Solution**:
- Reduce visualization DPI: Set `DPI=150` (default: 300)
- Process files individually instead of batch
- Increase system memory or use sampling

**Prevention**: Monitor memory usage with `--verbose` flag

#### 5. No Visualizations Generated
**Symptom**: Step completes successfully but no images created

**Diagnostic**:
```bash
# Check if GNN processing (step 3) completed successfully
ls output/3_gnn_output/

# Run visualization with verbose logging
python src/8_visualization.py --verbose --target-dir input/gnn_files --output-dir output
```

**Common Causes**:
- GNN processing (step 3) not run first
- Empty or invalid GNN files
- Missing parsed model files

**Solution**:
```bash
# Run complete pipeline in order
python src/main.py --only-steps "3,8" --verbose
```

#### 6. Visualization Quality Issues
**Symptom**: Blurry or pixelated visualizations

**Solution**:
- Increase DPI in configuration (default: 300)
- Use vector formats (SVG) instead of PNG
- Adjust figure size in config

**Configuration**:
```python
VISUALIZATION_CONFIG = {
    'dpi': 600,  # Higher quality
    'format': 'svg',  # Vector format
    'figsize': (12, 10)  # Larger canvas
}
```

#### 7. Progress Tracking Not Visible
**Symptom**: No progress updates during long-running visualizations

**Solution**:
```bash
# Enable verbose mode for detailed progress
python src/8_visualization.py --verbose --target-dir input/gnn_files
```

**Features**:
- ✅ File-by-file progress indicators: `[1/5]`, `[2/5]`, etc.
- ✅ Visualization type completion: Matrix ✅, Network ✅, Combined ✅
- ✅ Detailed step logging with emoji indicators 📊

### Performance Optimization

#### Fast Visualization Tips
1. **Use appropriate DPI**: 150 for preview, 300 for publication
2. **Sample large models**: Automatic sampling for >100 nodes
3. **Parallel processing**: Process multiple files independently
4. **Cache results**: Reuse visualizations when possible

#### Resource Management
- **Memory**: ~50-150MB per model (typical)
- **CPU**: 1-2 cores per visualization process
- **Disk**: ~1-5MB per visualization set
- **Time**: 1-5 seconds per model (typical)

### Best Practices

1. **Always run GNN processing (step 3) first**:
   ```bash
   python src/3_gnn.py --target-dir input/gnn_files
   python src/8_visualization.py --target-dir input/gnn_files
   ```

2. **Use verbose mode for debugging**:
   ```bash
   python src/8_visualization.py --verbose
   ```

3. **Check output directory structure**:
   ```
   output/8_visualization_output/
   ├── model_name/
   │   ├── matrix_analysis.png
   │   ├── matrix_statistics.png
   │   └── model_name_combined_analysis.png
   └── visualization_results.json
   ```

4. **Monitor for warnings**:
   - Backend configuration warnings
   - Dependency availability warnings
   - Sampling notifications for large models

---

## Version History

### Current Version: 1.0.0

**Features**:
- Graph visualization generation
- Matrix heatmap creation
- Network topology diagrams
- Model structure visualization
- Automatic headless environment detection
- Progress tracking with visual indicators

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced interactive visualizations
- **Future**: Real-time visualization updates

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Advanced Visualization](../advanced_visualization/AGENTS.md)
- [GNN Visualization Guide](../../doc/gnn/gnn_visualization.md)

### External Resources
- [Matplotlib Documentation](https://matplotlib.org/)
- [NetworkX Documentation](https://networkx.org/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Last Updated**: 2025-12-30
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern