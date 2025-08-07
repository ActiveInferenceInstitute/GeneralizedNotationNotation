# Advanced Visualization Module

This module provides comprehensive advanced visualization capabilities for GNN models, including interactive dashboards, 3D visualizations, and sophisticated data analysis visualizations.

## Module Structure

```
src/advanced_visualization/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── dashboard.py                   # Dashboard generation system
├── data_extractor.py              # Data extraction and processing
├── html_generator.py              # HTML visualization generation
└── visualizer.py                  # Main visualization orchestrator
```

## Core Components

### DashboardGenerator (`dashboard.py`)

Generates comprehensive interactive dashboards for GNN models.

#### Key Methods

- `generate_dashboard(content: str, model_name: str, output_dir: Path) -> Optional[Path]`
  - Creates a complete dashboard from GNN content
  - Returns path to generated dashboard HTML file
  - Handles strict validation and error recovery

- `_generate_dashboard_html(extracted_data: Dict[str, Any], model_name: str) -> str`
  - Generates HTML content for dashboard
  - Includes interactive components and styling
  - Provides comprehensive model analysis

#### Usage

```python
from advanced_visualization.dashboard import DashboardGenerator

generator = DashboardGenerator(strict_validation=True)
dashboard_path = generator.generate_dashboard(
    content=gnn_content,
    model_name="my_model",
    output_dir=Path("output/")
)
```

### VisualizationDataExtractor (`data_extractor.py`)

Extracts and processes data from GNN content for visualization.

#### Key Methods

- `extract_from_file(file_path: Path) -> Dict[str, Any]`
  - Extracts visualization data from GNN file
  - Returns structured data dictionary

- `extract_from_content(content: str, format_hint: Optional[GNNFormat] = None) -> Dict[str, Any]`
  - Extracts data from GNN content string
  - Supports multiple format hints

- `get_model_statistics(extracted_data: Dict[str, Any]) -> Dict[str, Any]`
  - Calculates comprehensive model statistics
  - Includes complexity metrics and structural analysis

#### Usage

```python
from advanced_visualization.data_extractor import VisualizationDataExtractor

extractor = VisualizationDataExtractor(strict_validation=True)
data = extractor.extract_from_content(gnn_content)
stats = extractor.get_model_statistics(data)
```

### HTMLVisualizationGenerator (`html_generator.py`)

Generates advanced HTML visualizations with interactive components.

#### Key Methods

- `generate_advanced_visualization(extracted_data: Dict[str, Any], model_name: str) -> str`
  - Creates comprehensive HTML visualization
  - Includes interactive charts and analysis
  - Provides error handling and fallback content

- `_generate_error_page(model_name: str, errors: List[str]) -> str`
  - Generates error page with diagnostic information
  - Provides recovery suggestions

#### Usage

```python
from advanced_visualization.html_generator import HTMLVisualizationGenerator

generator = HTMLVisualizationGenerator()
html_content = generator.generate_advanced_visualization(data, "model_name")
```

### AdvancedVisualizer (`visualizer.py`)

Main orchestrator for advanced visualization capabilities.

#### Key Methods

- `generate_visualizations(content: str, model_name: str, output_dir: Path, viz_type: str = "all", interactive: bool = True, export_formats: List[str] = None) -> List[str]`
  - Main method for generating all visualization types
  - Supports multiple visualization types and export formats
  - Returns list of generated file paths

- `_generate_3d_visualization(extracted_data: Dict[str, Any], model_name: str) -> str`
  - Creates 3D interactive visualizations
  - Uses Three.js for web-based 3D rendering

- `_generate_interactive_visualization(extracted_data: Dict[str, Any], model_name: str) -> str`
  - Generates interactive 2D visualizations
  - Includes zoom, pan, and selection capabilities

#### Usage

```python
from advanced_visualization.visualizer import AdvancedVisualizer

visualizer = AdvancedVisualizer(strict_validation=True)
generated_files = visualizer.generate_visualizations(
    content=gnn_content,
    model_name="my_model",
    output_dir=Path("output/"),
    viz_type="all",
    interactive=True
)
```

## Visualization Types

### 1. Interactive Dashboards
- Comprehensive model overview
- Real-time data exploration
- Interactive filtering and sorting
- Export capabilities

### 2. 3D Visualizations
- Three-dimensional model representation
- Interactive rotation and zoom
- Layer-based visualization
- Animation support

### 3. Network Graphs
- Interactive node-link diagrams
- Force-directed layouts
- Node clustering and grouping
- Edge weight visualization

### 4. Statistical Analysis
- Distribution plots
- Correlation matrices
- Time series analysis
- Performance metrics

### 5. Matrix Visualizations
- Heatmap representations
- Interactive matrix exploration
- Value highlighting
- Export to various formats

## Data Processing Pipeline

### 1. Content Extraction
```python
# Extract data from GNN content
extractor = VisualizationDataExtractor()
data = extractor.extract_from_content(gnn_content)
```

### 2. Statistical Analysis
```python
# Generate comprehensive statistics
stats = extractor.get_model_statistics(data)
```

### 3. Visualization Generation
```python
# Create visualizations
visualizer = AdvancedVisualizer()
files = visualizer.generate_visualizations(data, model_name, output_dir)
```

### 4. Dashboard Assembly
```python
# Generate complete dashboard
dashboard = DashboardGenerator()
dashboard_path = dashboard.generate_dashboard(content, model_name, output_dir)
```

## Error Handling and Recovery

### Fallback Mechanisms
- **Dependency Failures**: Graceful degradation to basic HTML
- **Data Extraction Errors**: Error pages with diagnostic information
- **Visualization Failures**: Alternative visualization methods
- **Export Failures**: Multiple export format attempts

### Error Reporting
```python
# Comprehensive error reporting
if not success:
    error_page = generator._generate_error_page(model_name, errors)
    # Save error page for debugging
```

## Performance Optimization

### Caching Strategies
- **Data Extraction**: Cache extracted data to avoid reprocessing
- **Visualization Generation**: Cache generated visualizations
- **Dashboard Assembly**: Incremental dashboard updates

### Memory Management
- **Large Models**: Streaming data processing for large models
- **Resource Cleanup**: Automatic cleanup of temporary files
- **Memory Monitoring**: Track memory usage during processing

## Integration with Pipeline

### Pipeline Step 9: Advanced Visualization
```python
# Called from 9_advanced_viz.py
def process_advanced_visualization(target_dir, output_dir, **kwargs):
    visualizer = AdvancedVisualizer()
    return visualizer.generate_visualizations(
        content=content,
        model_name=model_name,
        output_dir=output_dir
    )
```

### Output Structure
```
output/advanced_visualization/
├── dashboard.html                  # Main interactive dashboard
├── 3d_visualization.html          # 3D model visualization
├── network_graph.html             # Interactive network graph
├── statistics.html                # Statistical analysis
├── matrix_heatmap.html           # Matrix visualization
└── error_report.html             # Error diagnostics (if any)
```

## Configuration Options

### Visualization Settings
```python
# Configuration options
config = {
    'interactive': True,           # Enable interactive features
    'export_formats': ['html', 'png', 'svg'],  # Export formats
    'visualization_types': ['dashboard', '3d', 'network'],  # Viz types
    'strict_validation': True,     # Strict data validation
    'performance_mode': False      # Performance optimization
}
```

### Customization
```python
# Custom visualization parameters
visualizer = AdvancedVisualizer()
visualizer.set_custom_parameters({
    'color_scheme': 'viridis',
    'animation_speed': 1.0,
    'interaction_level': 'full'
})
```

## Testing and Validation

### Unit Tests
```python
# Test visualization generation
def test_visualization_generation():
    visualizer = AdvancedVisualizer()
    result = visualizer.generate_visualizations(test_content, "test", test_dir)
    assert len(result) > 0
```

### Integration Tests
```python
# Test pipeline integration
def test_pipeline_integration():
    success = process_advanced_visualization(test_dir, output_dir)
    assert success
```

## Dependencies

### Required Dependencies
- **matplotlib**: Basic plotting capabilities
- **networkx**: Network graph generation
- **numpy**: Numerical computations
- **pandas**: Data manipulation

### Optional Dependencies
- **plotly**: Interactive visualizations
- **bokeh**: Advanced interactive plots
- **three.js**: 3D visualizations (via HTML)
- **d3.js**: Data-driven documents (via HTML)

## Performance Metrics

### Processing Times
- **Small Models** (< 100 variables): < 1 second
- **Medium Models** (100-1000 variables): 1-5 seconds
- **Large Models** (> 1000 variables): 5-30 seconds

### Memory Usage
- **Base Memory**: ~50MB
- **Per Model**: ~10-100MB depending on complexity
- **Peak Memory**: 2-3x base usage during processing

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
```
Error: ModuleNotFoundError: No module named 'plotly'
Solution: Install optional dependencies or use fallback visualizations
```

#### 2. Memory Issues
```
Error: MemoryError during large model processing
Solution: Enable performance mode or process in chunks
```

#### 3. Visualization Failures
```
Error: Failed to generate 3D visualization
Solution: Check browser compatibility or use 2D fallback
```

### Debug Mode
```python
# Enable debug mode for detailed logging
visualizer = AdvancedVisualizer(debug=True)
```

## Future Enhancements

### Planned Features
- **Real-time Updates**: Live visualization updates
- **Collaborative Features**: Multi-user visualization sessions
- **Advanced Analytics**: Machine learning-based insights
- **Mobile Support**: Responsive design for mobile devices

### Performance Improvements
- **WebGL Rendering**: Hardware-accelerated 3D rendering
- **Streaming Processing**: Real-time data streaming
- **Caching Optimization**: Advanced caching strategies

## Summary

The Advanced Visualization module provides comprehensive visualization capabilities for GNN models, including interactive dashboards, 3D visualizations, and sophisticated data analysis. The module is designed with robust error handling, performance optimization, and extensive customization options to support various use cases in Active Inference research and development.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md