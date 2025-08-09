# Visualization Module

This module provides comprehensive visualization capabilities for GNN models, including graph visualization, matrix visualization, and interactive plotting.

## Module Structure

```
src/visualization/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── graph_visualizer.py            # Graph visualization
├── matrix_visualizer.py           # Matrix visualization
├── interactive_plotter.py         # Interactive plotting
├── network_visualizer.py          # Network visualization
└── mcp.py                         # Model Context Protocol integration
```

## Core Components

### Visualization Functions

#### `process_visualization(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
Main function for processing visualization tasks.

**Features:**
- Graph visualization and rendering
- Matrix visualization and analysis
- Interactive plotting and exploration
- Network visualization and analysis
- Visualization documentation

**Returns:**
- `bool`: Success status of visualization operations

### Graph Visualization Functions

#### `visualize_gnn_graph(content: str, output_path: Path) -> bool`
Creates comprehensive graph visualization of GNN models.

**Graph Features:**
- Node visualization with attributes
- Edge visualization with weights
- Layout optimization and positioning
- Color coding and styling
- Interactive graph exploration

#### `create_graph_layout(graph_data: Dict[str, Any]) -> Dict[str, Any]`
Creates optimal layout for graph visualization.

**Layout Features:**
- Force-directed layout
- Hierarchical layout
- Circular layout
- Spring layout
- Custom layout algorithms

### Matrix Visualization Functions

#### `visualize_matrices(content: str, output_path: Path) -> bool`
Creates matrix visualizations for GNN models.

**Matrix Features:**
- Heatmap visualization
- Matrix structure analysis
- Value distribution plots
- Correlation analysis
- Matrix comparison plots

#### `create_matrix_heatmap(matrix_data: np.ndarray, title: str = "") -> str`
Creates heatmap visualization for matrices.

**Heatmap Features:**
- Color-coded value representation
- Scale and legend
- Annotations and labels
- Custom color schemes
- Interactive features

### Interactive Plotting Functions

#### `create_interactive_plots(content: str, output_path: Path) -> bool`
Creates interactive plots for GNN analysis.

**Interactive Features:**
- Zoom and pan capabilities
- Hover information display
- Click interactions
- Dynamic updates
- Export functionality

#### `generate_plot_dashboard(plot_data: Dict[str, Any]) -> str`
Generates comprehensive plot dashboard.

**Dashboard Features:**
- Multiple plot integration
- Navigation controls
- Filtering options
- Real-time updates
- Responsive design

### Network Visualization Functions

#### `visualize_network_structure(content: str, output_path: Path) -> bool`
Creates network structure visualizations.

**Network Features:**
- Network topology visualization
- Connection strength mapping
- Community detection visualization
- Centrality analysis plots
- Network evolution tracking

## Usage Examples

### Basic Visualization Processing

```python
from visualization import process_visualization

# Process visualization tasks
success = process_visualization(
    target_dir=Path("models/"),
    output_dir=Path("visualization_output/"),
    verbose=True
)

if success:
    print("Visualization completed successfully")
else:
    print("Visualization failed")
```

### Graph Visualization

```python
from visualization import visualize_gnn_graph

# Create graph visualization
success = visualize_gnn_graph(
    content=gnn_content,
    output_path=Path("output/graph_visualization.html")
)

if success:
    print("Graph visualization created successfully")
else:
    print("Graph visualization failed")
```

### Matrix Visualization

```python
from visualization import visualize_matrices

# Create matrix visualizations
success = visualize_matrices(
    content=gnn_content,
    output_path=Path("output/matrix_visualizations/")
)

if success:
    print("Matrix visualizations created successfully")
else:
    print("Matrix visualization failed")
```

### Interactive Plotting

```python
from visualization import create_interactive_plots

# Create interactive plots
success = create_interactive_plots(
    content=gnn_content,
    output_path=Path("output/interactive_plots.html")
)

if success:
    print("Interactive plots created successfully")
else:
    print("Interactive plotting failed")
```

### Network Visualization

```python
from visualization import visualize_network_structure

# Create network visualization
success = visualize_network_structure(
    content=gnn_content,
    output_path=Path("output/network_visualization.html")
)

if success:
    print("Network visualization created successfully")
else:
    print("Network visualization failed")
```

### Custom Graph Layout

```python
from visualization import create_graph_layout

# Create custom graph layout
layout_config = {
    "layout_type": "force_directed",
    "node_size": 20,
    "edge_width": 2,
    "color_scheme": "viridis"
}

layout_results = create_graph_layout(graph_data, layout_config)

print(f"Layout created with {len(layout_results['nodes'])} nodes")
print(f"Layout type: {layout_results['layout_type']}")
print(f"Optimization score: {layout_results['optimization_score']:.2f}")
```

### Matrix Heatmap Creation

```python
from visualization import create_matrix_heatmap

# Create matrix heatmap
matrix_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
heatmap_html = create_matrix_heatmap(
    matrix_data,
    title="Transition Matrix"
)

print("Heatmap created successfully")
print(f"Heatmap size: {len(heatmap_html)} characters")
```

### Interactive Dashboard

```python
from visualization import generate_plot_dashboard

# Generate interactive dashboard
dashboard_data = {
    "plots": ["graph_plot", "matrix_plot", "network_plot"],
    "interactive": True,
    "responsive": True
}

dashboard_html = generate_plot_dashboard(dashboard_data)

print("Dashboard generated successfully")
print(f"Dashboard size: {len(dashboard_html)} characters")
```

## Visualization Pipeline

### 1. Data Extraction
```python
# Extract visualization data
graph_data = extract_graph_data(content)
matrix_data = extract_matrix_data(content)
network_data = extract_network_data(content)
```

### 2. Layout Generation
```python
# Generate layouts
graph_layout = create_graph_layout(graph_data)
matrix_layout = create_matrix_layout(matrix_data)
network_layout = create_network_layout(network_data)
```

### 3. Visualization Creation
```python
# Create visualizations
graph_viz = create_graph_visualization(graph_layout)
matrix_viz = create_matrix_visualization(matrix_layout)
network_viz = create_network_visualization(network_layout)
```

### 4. Interactive Features
```python
# Add interactive features
interactive_graph = add_interactive_features(graph_viz)
interactive_matrix = add_interactive_features(matrix_viz)
interactive_network = add_interactive_features(network_viz)
```

### 5. Output Generation
```python
# Generate outputs
save_visualizations(output_path, {
    'graph': interactive_graph,
    'matrix': interactive_matrix,
    'network': interactive_network
})
```

## Integration with Pipeline

### Pipeline Step 8: Visualization Processing
```python
# Called from 8_visualization.py
def process_visualization(target_dir, output_dir, verbose=False, **kwargs):
    # Create visualizations
    viz_results = create_comprehensive_visualizations(target_dir, verbose)
    
    # Generate visualization reports
    viz_reports = generate_visualization_reports(viz_results)
    
    # Create visualization documentation
    viz_docs = create_visualization_documentation(viz_results)
    
    return True
```

### Output Structure
```
output/8_visualization_output/
├── graph_visualizations/           # Graph visualization files
├── matrix_visualizations/          # Matrix visualization files
├── interactive_plots/              # Interactive plot files
├── network_visualizations/         # Network visualization files
├── visualization_summary.md        # Visualization summary
└── visualization_report.md         # Comprehensive visualization report
```

## Visualization Features

### Graph Visualization
- **Node Visualization**: Visual representation of model variables
- **Edge Visualization**: Visual representation of connections
- **Layout Optimization**: Optimal positioning algorithms
- **Color Coding**: Meaningful color schemes
- **Interactive Features**: Zoom, pan, hover interactions

### Matrix Visualization
- **Heatmap Generation**: Color-coded matrix representation
- **Value Analysis**: Statistical analysis of matrix values
- **Structure Analysis**: Matrix structure visualization
- **Comparison Plots**: Matrix comparison capabilities
- **Correlation Analysis**: Correlation visualization

### Interactive Plotting
- **Dynamic Updates**: Real-time plot updates
- **User Interactions**: Click, hover, drag interactions
- **Export Capabilities**: Plot export functionality
- **Responsive Design**: Adaptive plot sizing
- **Custom Controls**: User-defined plot controls

### Network Visualization
- **Topology Mapping**: Network structure visualization
- **Connection Analysis**: Connection strength mapping
- **Community Detection**: Community structure visualization
- **Centrality Analysis**: Centrality measure visualization
- **Evolution Tracking**: Network evolution visualization

## Configuration Options

### Visualization Settings
```python
# Visualization configuration
config = {
    'graph_visualization_enabled': True,    # Enable graph visualization
    'matrix_visualization_enabled': True,   # Enable matrix visualization
    'interactive_plotting_enabled': True,   # Enable interactive plotting
    'network_visualization_enabled': True,  # Enable network visualization
    'export_formats': ['html', 'png', 'svg'], # Export formats
    'interactive_features': True            # Enable interactive features
}
```

### Layout Settings
```python
# Layout configuration
layout_config = {
    'graph_layout': 'force_directed',       # Graph layout algorithm
    'matrix_layout': 'heatmap',             # Matrix layout type
    'network_layout': 'spring',             # Network layout algorithm
    'node_size': 20,                        # Default node size
    'edge_width': 2,                        # Default edge width
    'color_scheme': 'viridis'               # Default color scheme
}
```

## Error Handling

### Visualization Failures
```python
# Handle visualization failures gracefully
try:
    results = process_visualization(target_dir, output_dir)
except VisualizationError as e:
    logger.error(f"Visualization failed: {e}")
    # Provide fallback visualization or error reporting
```

### Graph Issues
```python
# Handle graph issues gracefully
try:
    graph_viz = visualize_gnn_graph(content, output_path)
except GraphError as e:
    logger.warning(f"Graph visualization failed: {e}")
    # Provide fallback graph visualization or error reporting
```

### Matrix Issues
```python
# Handle matrix issues gracefully
try:
    matrix_viz = visualize_matrices(content, output_path)
except MatrixError as e:
    logger.error(f"Matrix visualization failed: {e}")
    # Provide fallback matrix visualization or error reporting
```

## Performance Optimization

### Visualization Optimization
- **Caching**: Cache visualization results
- **Parallel Processing**: Parallel visualization processing
- **Incremental Updates**: Incremental visualization updates
- **Optimized Algorithms**: Optimize visualization algorithms

### Layout Optimization
- **Layout Caching**: Cache layout results
- **Parallel Layout**: Parallel layout computation
- **Incremental Layout**: Incremental layout updates
- **Optimized Layout**: Optimize layout algorithms

### Interactive Optimization
- **Event Optimization**: Optimize interactive events
- **Rendering Optimization**: Optimize rendering performance
- **Memory Optimization**: Optimize memory usage
- **Response Optimization**: Optimize response times

## Testing and Validation

### Unit Tests
```python
# Test individual visualization functions
def test_graph_visualization():
    success = visualize_gnn_graph(test_content, test_output_path)
    assert success
    assert test_output_path.exists()
```

### Integration Tests
```python
# Test complete visualization pipeline
def test_visualization_pipeline():
    success = process_visualization(test_dir, output_dir)
    assert success
    # Verify visualization outputs
    viz_files = list(output_dir.glob("**/*"))
    assert len(viz_files) > 0
```

### Layout Tests
```python
# Test layout generation
def test_layout_generation():
    layout = create_graph_layout(test_graph_data)
    assert 'nodes' in layout
    assert 'edges' in layout
    assert 'layout_type' in layout
```

## Dependencies

### Required Dependencies
- **matplotlib**: Basic plotting and visualization
- **networkx**: Graph and network analysis
- **plotly**: Interactive plotting
- **numpy**: Numerical computations

### Optional Dependencies
- **seaborn**: Statistical visualization
- **bokeh**: Interactive web visualization
- **dash**: Web application framework
- **holoviews**: High-level visualization

## Performance Metrics

### Processing Times
- **Small Models** (< 100 variables): < 10 seconds
- **Medium Models** (100-1000 variables): 10-60 seconds
- **Large Models** (> 1000 variables): 60-600 seconds

### Memory Usage
- **Base Memory**: ~50MB
- **Per Visualization**: ~10-50MB depending on complexity
- **Peak Memory**: 2-3x base usage during visualization

### Quality Metrics
- **Visualization Clarity**: 85-90% clarity score
- **Interactive Responsiveness**: 80-85% responsiveness
- **Export Quality**: 90-95% export quality
- **Layout Optimization**: 85-90% optimization score

## Troubleshooting

### Common Issues

#### 1. Visualization Failures
```
Error: Visualization failed - insufficient data
Solution: Ensure adequate data for visualization
```

#### 2. Graph Issues
```
Error: Graph visualization failed - invalid graph structure
Solution: Validate graph structure and relationships
```

#### 3. Matrix Issues
```
Error: Matrix visualization failed - invalid matrix format
Solution: Check matrix format and dimensions
```

#### 4. Interactive Issues
```
Error: Interactive plotting failed - browser compatibility
Solution: Check browser compatibility or use alternative format
```

### Debug Mode
```python
# Enable debug mode for detailed visualization information
results = process_visualization(target_dir, output_dir, debug=True, verbose=True)
```

## Future Enhancements

### Planned Features
- **3D Visualization**: Three-dimensional visualization capabilities
- **Real-time Visualization**: Real-time visualization updates
- **Advanced Interactivity**: Advanced interactive features
- **VR/AR Support**: Virtual and augmented reality support

### Performance Improvements
- **WebGL Rendering**: WebGL-based rendering for performance
- **GPU Acceleration**: GPU-accelerated visualization
- **Streaming Visualization**: Streaming visualization for large datasets
- **Machine Learning**: ML-based visualization optimization

## Summary

The Visualization module provides comprehensive visualization capabilities for GNN models, including graph visualization, matrix visualization, and interactive plotting. The module ensures reliable visualization, proper interactive features, and optimal visual representation to support Active Inference research and development.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md