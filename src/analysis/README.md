# Analysis Module

This module provides comprehensive statistical analysis, performance profiling, and model evaluation capabilities for GNN models and pipeline components.

## Module Structure

```
src/analysis/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
└── mcp.py                         # Model Context Protocol integration
```

## Core Components

### Statistical Analysis Functions

#### `perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
Performs comprehensive statistical analysis on GNN model files.

**Features:**
- Variable distribution analysis
- Connection pattern analysis
- Complexity metrics calculation
- Performance benchmarking
- Model comparison capabilities

**Returns:**
- Dictionary containing comprehensive analysis results
- Statistical summaries and metrics
- Performance benchmarks
- Model comparison data

#### `extract_variables_for_analysis(content: str) -> List[Dict[str, Any]]`
Extracts and analyzes variables from GNN content.

**Features:**
- Variable type classification
- Dimension analysis
- Data type validation
- Complexity assessment

#### `extract_connections_for_analysis(content: str) -> List[Dict[str, Any]]`
Extracts and analyzes connections from GNN content.

**Features:**
- Connection pattern analysis
- Dependency mapping
- Graph structure analysis
- Connectivity metrics

#### `extract_sections_for_analysis(content: str) -> List[Dict[str, Any]]`
Extracts and analyzes GNN sections for comprehensive analysis.

**Features:**
- Section type classification
- Content structure analysis
- Semantic analysis
- Validation metrics

### Statistical Calculation Functions

#### `calculate_variable_statistics(variables: List[Dict[str, Any]]) -> Dict[str, Any]`
Calculates comprehensive statistics for variables.

**Metrics:**
- Type distribution
- Dimension statistics
- Complexity measures
- Memory usage estimates

#### `calculate_connection_statistics(connections: List[Dict[str, Any]]) -> Dict[str, Any]`
Calculates statistics for model connections.

**Metrics:**
- Connection density
- Graph metrics
- Dependency patterns
- Structural complexity

#### `calculate_section_statistics(sections: List[Dict[str, Any]]) -> Dict[str, Any]`
Calculates statistics for GNN sections.

**Metrics:**
- Section distribution
- Content analysis
- Validation status
- Quality metrics

### Complexity Analysis Functions

#### `calculate_cyclomatic_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float`
Calculates cyclomatic complexity of the model.

**Formula:**
```
Complexity = E - N + 2P
Where:
- E = Number of edges (connections)
- N = Number of nodes (variables)
- P = Number of connected components
```

#### `calculate_cognitive_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float`
Calculates cognitive complexity based on model structure.

**Factors:**
- Variable type diversity
- Connection patterns
- Nesting levels
- Semantic complexity

#### `calculate_structural_complexity(variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float`
Calculates structural complexity metrics.

**Metrics:**
- Graph density
- Clustering coefficient
- Path length analysis
- Modularity measures

### Performance Analysis Functions

#### `run_performance_benchmarks(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
Runs comprehensive performance benchmarks.

**Benchmarks:**
- Processing time analysis
- Memory usage profiling
- CPU utilization
- I/O performance
- Scalability testing

#### `calculate_complexity_metrics(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
Calculates comprehensive complexity metrics.

**Metrics:**
- Cyclomatic complexity
- Cognitive complexity
- Structural complexity
- Maintainability index
- Technical debt assessment

### Quality Assessment Functions

#### `calculate_maintainability_index(content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float`
Calculates maintainability index for the model.

**Formula:**
```
MI = 171 - 5.2 * ln(HV) - 0.23 * ln(CC) - 16.2 * ln(LOC)
Where:
- HV = Halstead Volume
- CC = Cyclomatic Complexity
- LOC = Lines of Code
```

#### `calculate_technical_debt(content: str, variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]) -> float`
Calculates technical debt for the model.

**Factors:**
- Code quality issues
- Complexity penalties
- Documentation gaps
- Testing coverage
- Performance bottlenecks

### Model Comparison Functions

#### `perform_model_comparisons(statistical_analyses: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]`
Performs comparative analysis across multiple models.

**Comparisons:**
- Performance benchmarking
- Complexity comparison
- Quality assessment
- Feature analysis
- Best practices evaluation

### Reporting Functions

#### `generate_analysis_summary(results: Dict[str, Any]) -> str`
Generates comprehensive analysis summary.

**Content:**
- Executive summary
- Key metrics
- Recommendations
- Risk assessment
- Improvement suggestions

## Usage Examples

### Basic Statistical Analysis

```python
from analysis import perform_statistical_analysis

# Analyze a GNN model file
results = perform_statistical_analysis(
    file_path=Path("models/my_model.md"),
    verbose=True
)

print(f"Model complexity: {results['complexity_metrics']['cyclomatic']}")
print(f"Variable count: {results['statistics']['variable_count']}")
print(f"Connection count: {results['statistics']['connection_count']}")
```

### Comprehensive Analysis

```python
from analysis import (
    extract_variables_for_analysis,
    extract_connections_for_analysis,
    calculate_variable_statistics,
    calculate_connection_statistics
)

# Extract and analyze components
variables = extract_variables_for_analysis(gnn_content)
connections = extract_connections_for_analysis(gnn_content)

# Calculate statistics
var_stats = calculate_variable_statistics(variables)
conn_stats = calculate_connection_statistics(connections)

print(f"Variable types: {var_stats['type_distribution']}")
print(f"Connection density: {conn_stats['density']}")
```

### Performance Benchmarking

```python
from analysis import run_performance_benchmarks

# Run performance benchmarks
benchmarks = run_performance_benchmarks(
    file_path=Path("models/large_model.md"),
    verbose=True
)

print(f"Processing time: {benchmarks['processing_time']:.3f}s")
print(f"Memory usage: {benchmarks['memory_usage']:.2f}MB")
print(f"CPU utilization: {benchmarks['cpu_utilization']:.1f}%")
```

### Complexity Analysis

```python
from analysis import (
    calculate_cyclomatic_complexity,
    calculate_cognitive_complexity,
    calculate_structural_complexity
)

# Calculate complexity metrics
cyclomatic = calculate_cyclomatic_complexity(variables, connections)
cognitive = calculate_cognitive_complexity(variables, connections)
structural = calculate_structural_complexity(variables, connections)

print(f"Cyclomatic complexity: {cyclomatic:.2f}")
print(f"Cognitive complexity: {cognitive:.2f}")
print(f"Structural complexity: {structural:.2f}")
```

### Quality Assessment

```python
from analysis import (
    calculate_maintainability_index,
    calculate_technical_debt
)

# Assess model quality
maintainability = calculate_maintainability_index(content, variables, connections)
tech_debt = calculate_technical_debt(content, variables, connections)

print(f"Maintainability index: {maintainability:.2f}")
print(f"Technical debt: {tech_debt:.2f}")
```

## Analysis Pipeline

### 1. Data Extraction
```python
# Extract model components
variables = extract_variables_for_analysis(content)
connections = extract_connections_for_analysis(content)
sections = extract_sections_for_analysis(content)
```

### 2. Statistical Analysis
```python
# Calculate comprehensive statistics
var_stats = calculate_variable_statistics(variables)
conn_stats = calculate_connection_statistics(connections)
section_stats = calculate_section_statistics(sections)
```

### 3. Complexity Assessment
```python
# Assess model complexity
complexity_metrics = {
    'cyclomatic': calculate_cyclomatic_complexity(variables, connections),
    'cognitive': calculate_cognitive_complexity(variables, connections),
    'structural': calculate_structural_complexity(variables, connections)
}
```

### 4. Performance Evaluation
```python
# Evaluate performance characteristics
performance = run_performance_benchmarks(file_path)
```

### 5. Quality Assessment
```python
# Assess model quality
quality_metrics = {
    'maintainability': calculate_maintainability_index(content, variables, connections),
    'technical_debt': calculate_technical_debt(content, variables, connections)
}
```

## Integration with Pipeline

### Pipeline Step 16: Analysis
```python
# Called from 16_analysis.py
def process_analysis(target_dir, output_dir, verbose=False, **kwargs):
    # Perform comprehensive analysis
    results = perform_statistical_analysis(file_path, verbose)
    
    # Generate analysis report
    summary = generate_analysis_summary(results)
    
    # Save results
    save_analysis_results(results, output_dir)
    
    return True
```

### Output Structure
```
output/analysis/
├── statistical_analysis.json       # Comprehensive analysis results
├── performance_benchmarks.json     # Performance metrics
├── complexity_metrics.json         # Complexity analysis
├── quality_assessment.json         # Quality metrics
├── model_comparison.json          # Comparative analysis
└── analysis_summary.md            # Human-readable summary
```

## Analysis Metrics

### Statistical Metrics
- **Variable Count**: Total number of variables
- **Connection Count**: Total number of connections
- **Type Distribution**: Distribution of variable types
- **Dimension Analysis**: Variable dimension statistics
- **Density Metrics**: Connection density and patterns

### Complexity Metrics
- **Cyclomatic Complexity**: Graph-based complexity measure
- **Cognitive Complexity**: Human comprehension difficulty
- **Structural Complexity**: Model structure complexity
- **Maintainability Index**: Code maintainability score
- **Technical Debt**: Quality and maintainability debt

### Performance Metrics
- **Processing Time**: Model processing duration
- **Memory Usage**: Memory consumption during processing
- **CPU Utilization**: CPU usage patterns
- **I/O Performance**: Input/output performance
- **Scalability**: Performance scaling characteristics

### Quality Metrics
- **Code Quality**: Overall code quality assessment
- **Documentation Coverage**: Documentation completeness
- **Testing Coverage**: Test coverage metrics
- **Best Practices**: Adherence to best practices
- **Risk Assessment**: Potential risk factors

## Configuration Options

### Analysis Settings
```python
# Configuration options
config = {
    'verbose': True,               # Enable detailed logging
    'include_performance': True,   # Include performance analysis
    'include_complexity': True,    # Include complexity analysis
    'include_quality': True,       # Include quality assessment
    'benchmark_iterations': 5,     # Number of benchmark iterations
    'memory_profiling': True,      # Enable memory profiling
    'cpu_profiling': True          # Enable CPU profiling
}
```

### Custom Metrics
```python
# Define custom analysis metrics
custom_metrics = {
    'custom_complexity': lambda v, c: custom_complexity_calculation(v, c),
    'custom_quality': lambda content, v, c: custom_quality_assessment(content, v, c)
}
```

## Error Handling

### Analysis Failures
```python
# Handle analysis failures gracefully
try:
    results = perform_statistical_analysis(file_path)
except AnalysisError as e:
    logger.error(f"Analysis failed: {e}")
    # Provide fallback analysis or error reporting
```

### Data Validation
```python
# Validate input data before analysis
if not validate_gnn_content(content):
    raise ValueError("Invalid GNN content for analysis")
```

## Performance Considerations

### Optimization Strategies
- **Caching**: Cache analysis results for repeated analysis
- **Parallel Processing**: Use parallel processing for large models
- **Memory Management**: Optimize memory usage for large datasets
- **Incremental Analysis**: Support incremental analysis for large models

### Scalability
- **Large Models**: Handle models with thousands of variables
- **Batch Processing**: Process multiple models efficiently
- **Resource Management**: Manage CPU and memory resources
- **Progress Tracking**: Track analysis progress for long-running operations

## Testing and Validation

### Unit Tests
```python
# Test individual analysis functions
def test_variable_statistics():
    variables = extract_variables_for_analysis(test_content)
    stats = calculate_variable_statistics(variables)
    assert 'type_distribution' in stats
    assert 'count' in stats
```

### Integration Tests
```python
# Test complete analysis pipeline
def test_analysis_pipeline():
    results = perform_statistical_analysis(test_file)
    assert 'statistics' in results
    assert 'complexity_metrics' in results
    assert 'performance_benchmarks' in results
```

## Dependencies

### Required Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **networkx**: Graph analysis and metrics
- **matplotlib**: Statistical plotting
- **scipy**: Statistical functions

### Optional Dependencies
- **psutil**: System resource monitoring
- **memory_profiler**: Memory usage profiling
- **line_profiler**: Line-by-line profiling

## Performance Metrics

### Processing Times
- **Small Models** (< 100 variables): < 0.1 seconds
- **Medium Models** (100-1000 variables): 0.1-1.0 seconds
- **Large Models** (> 1000 variables): 1.0-10.0 seconds

### Memory Usage
- **Base Memory**: ~20MB
- **Per Model**: ~5-50MB depending on complexity
- **Peak Memory**: 1.5-2x base usage during analysis

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```
Error: MemoryError during large model analysis
Solution: Enable memory optimization or process in chunks
```

#### 2. Performance Issues
```
Error: Analysis taking too long for large models
Solution: Enable parallel processing or use sampling
```

#### 3. Data Validation Issues
```
Error: Invalid GNN content for analysis
Solution: Validate input data before analysis
```

### Debug Mode
```python
# Enable debug mode for detailed analysis
results = perform_statistical_analysis(file_path, verbose=True, debug=True)
```

## Future Enhancements

### Planned Features
- **Machine Learning Analysis**: ML-based model assessment
- **Predictive Analytics**: Performance prediction capabilities
- **Real-time Analysis**: Live analysis during model development
- **Advanced Visualizations**: Interactive analysis visualizations

### Performance Improvements
- **GPU Acceleration**: GPU-accelerated analysis for large models
- **Distributed Processing**: Distributed analysis for very large models
- **Streaming Analysis**: Real-time streaming analysis capabilities

## Summary

The Analysis module provides comprehensive statistical analysis, performance profiling, and model evaluation capabilities for GNN models. The module includes sophisticated complexity metrics, quality assessment tools, and performance benchmarking capabilities to support Active Inference research and development.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md