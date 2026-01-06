# Analysis Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced statistical analysis, performance benchmarking, and complexity metrics calculation for GNN models

**Pipeline Step**: Step 16: Analysis (16_analysis.py)

**Category**: Statistical Analysis / Performance Evaluation

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

---

## Core Functionality

### Primary Responsibilities
1. Perform comprehensive statistical analysis on GNN model structures
2. Calculate complexity metrics and maintainability indices
3. Generate performance benchmarks and comparison reports
4. Extract and analyze variable distributions and correlations
5. Provide technical debt assessment and optimization recommendations

### Key Capabilities
- Statistical analysis of model variables and connections
- Complexity metrics calculation (cyclomatic, cognitive, structural)
- Performance benchmarking and profiling
- Model comparison and differential analysis
- Distribution analysis and correlation studies

---

## API Reference

### Public Functions

#### `process_analysis(target_dir, output_dir, logger=None, **kwargs) -> bool`
**Description**: Main analysis processing function called by orchestrator (16_analysis.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for analysis results
- `logger` (Logger, optional): Logger instance for progress reporting (default: None)
- `analysis_type` (str): Type of analysis to perform ("comprehensive", "statistical", "performance", default: "comprehensive")
- `include_performance` (bool): Include performance benchmarking (default: True)
- `**kwargs`: Additional analysis options

**Returns**: `True` if analysis succeeded

**Example**:
```python
from analysis import process_analysis

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    analysis_type="comprehensive",
    include_performance=True
)
```

#### `perform_statistical_analysis(variables, connections) -> Dict[str, Any]`
**Description**: Perform comprehensive statistical analysis on model components

**Parameters**:
- `variables` (List[Dict]): Model variables data
- `connections` (List[Dict]): Model connections data

**Returns**: Dictionary with statistical analysis results

#### `calculate_complexity_metrics(model_data) -> Dict[str, Any]`
**Description**: Calculate various complexity metrics for GNN models

**Parameters**:
- `model_data` (Dict): Parsed GNN model data

**Returns**: Dictionary with complexity metrics (cyclomatic, cognitive, structural)

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations and statistical analysis
- `pandas` - Data manipulation and analysis
- `scipy` - Advanced statistical functions

### Optional Dependencies
- `matplotlib` - Statistical visualization (fallback: text-based reports)
- `seaborn` - Enhanced statistical plots (fallback: matplotlib)

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing patterns
- `pipeline.config` - Pipeline configuration management

---

## Configuration

### Environment Variables
- `ANALYSIS_PERFORMANCE_MODE` - Performance analysis mode ("fast", "comprehensive")
- `ANALYSIS_TIMEOUT` - Maximum analysis time per model (default: 300 seconds)

### Configuration Files
- `analysis_config.yaml` - Custom analysis parameters and thresholds

### Default Settings
```python
DEFAULT_COMPLEXITY_THRESHOLDS = {
    'cyclomatic_complexity': {'low': 10, 'medium': 20, 'high': 50},
    'cognitive_complexity': {'low': 5, 'medium': 15, 'high': 35},
    'structural_complexity': {'low': 100, 'medium': 500, 'high': 1000}
}
```

---

## Usage Examples

### Basic Usage
```python
from analysis.processor import process_analysis

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    logger=logger,
    analysis_type="comprehensive"
)
```

### Statistical Analysis
```python
from analysis.analyzer import perform_statistical_analysis

stats = perform_statistical_analysis(variables, connections)
print(f"Variable count: {stats['variable_statistics']['count']}")
print(f"Connection density: {stats['connection_statistics']['density']}")
```

### Complexity Assessment
```python
from analysis.analyzer import calculate_complexity_metrics

metrics = calculate_complexity_metrics(parsed_model)
print(f"Cyclomatic complexity: {metrics['cyclomatic_complexity']}")
print(f"Maintainability index: {metrics['maintainability_index']}")
```

---

## Output Specification

### Output Products
- `{model}_statistical_analysis.json` - Comprehensive statistical analysis
- `{model}_complexity_metrics.json` - Complexity assessment results
- `{model}_performance_benchmarks.json` - Performance profiling data
- `{model}_analysis_summary.md` - Human-readable analysis report
- `analysis_processing_summary.json` - Pipeline step summary

### Output Directory Structure
```
output/16_analysis_output/
├── model_name_statistical_analysis.json
├── model_name_complexity_metrics.json
├── model_name_performance_benchmarks.json
├── model_name_analysis_summary.md
└── analysis_processing_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 seconds per model
- **Memory**: ~50-100MB for large models
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Path**: ~1-2s for basic statistical analysis
- **Slow Path**: ~5-10s for comprehensive complexity analysis
- **Memory**: ~20-50MB for typical models, ~100MB for large models

---

## Error Handling

### Graceful Degradation
- **No scipy**: Simplified statistical analysis using numpy
- **No matplotlib**: Text-based statistical reports
- **Large models**: Sampling-based analysis with warnings

### Error Categories
1. **Statistical Errors**: Invalid data types or missing values
2. **Complexity Errors**: Model structure too complex for analysis
3. **Performance Errors**: Timeout or resource exhaustion

---

## Integration Points

### Orchestrated By
- **Script**: `16_analysis.py` (Step 16)
- **Function**: `process_analysis()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_analysis_integration.py` - Integration tests
- `report.generator` - Report generation uses analysis results

### Data Flow
```
GNN Files → Analysis → Statistical Reports → Model Comparisons → Optimization Recommendations
```

---

## Testing

### Test Files
- `src/tests/test_analysis_integration.py` - Integration tests
- `src/tests/test_analysis_unit.py` - Unit tests
- `src/tests/test_analysis_performance.py` - Performance tests

### Test Coverage
- **Current**: 80%
- **Target**: 90%+

### Key Test Scenarios
1. Statistical analysis with various model sizes
2. Complexity metric calculation accuracy
3. Performance benchmarking under load
4. Error handling with malformed data

---

## MCP Integration

### Tools Registered
- `analysis_perform` - Perform statistical analysis
- `analysis_complexity` - Calculate complexity metrics
- `analysis_compare` - Compare multiple models

### Tool Endpoints
```python
@mcp_tool("analysis_perform")
def perform_analysis_tool(model_data, analysis_type="comprehensive"):
    """Perform statistical analysis on GNN model"""
    # Implementation
```

---
