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

#### `process_analysis(target_dir: Path, output_dir: Path, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main analysis processing function called by orchestrator (16_analysis.py). Performs comprehensive statistical analysis, complexity metrics, and performance benchmarking.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for analysis results
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `analysis_type` (str, optional): Type of analysis ("comprehensive", "statistical", "performance", "complexity") (default: "comprehensive")
- `include_performance` (bool, optional): Include performance benchmarking (default: True)
- `include_complexity` (bool, optional): Include complexity metrics (default: True)
- `include_quality` (bool, optional): Include quality assessment (default: True)
- `benchmark_iterations` (int, optional): Number of benchmark iterations (default: 5)
- `**kwargs`: Additional analysis options

**Returns**: `bool` - True if analysis succeeded, False otherwise

**Example**:
```python
from analysis import process_analysis
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    logger=logger,
    analysis_type="comprehensive",
    include_performance=True,
    benchmark_iterations=10
)
```

#### `perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Perform comprehensive statistical analysis on a GNN file.

**Parameters**:
- `file_path` (Path): Path to the GNN file to analyze
- `verbose` (bool, optional): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Statistical analysis results with:
- `variable_count` (int): Total number of variables
- `connection_count` (int): Total number of connections
- `type_distribution` (Dict[str, int]): Distribution of variable types
- `dimension_statistics` (Dict[str, Any]): Dimension statistics
- `density_metrics` (Dict[str, float]): Connection density metrics

#### `calculate_complexity_metrics(model_data: Dict[str, Any], variables: List[Dict[str, Any]] = None, connections: List[Dict[str, Any]] = None) -> Dict[str, Any]`
**Description**: Calculate various complexity metrics for GNN models.

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data
- `variables` (List[Dict[str, Any]], optional): Model variables (extracted if not provided)
- `connections` (List[Dict[str, Any]], optional): Model connections (extracted if not provided)

**Returns**: `Dict[str, Any]` - Complexity metrics with:
- `cyclomatic_complexity` (float): Cyclomatic complexity score
- `cognitive_complexity` (float): Cognitive complexity score
- `structural_complexity` (float): Structural complexity score
- `maintainability_index` (float): Maintainability index (0-100)
- `technical_debt` (float): Technical debt score

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
- `process_analysis` - Process analysis for GNN files in a directory

### Tool Endpoints
```python
@mcp_tool("process_analysis")
def process_analysis_mcp(target_directory: str, output_directory: str, verbose: bool = False):
    """Process Analysis for GNN files. Exposed via MCP."""
    # Implementation
```

### MCP File Location
- `src/analysis/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Analysis fails on large models
**Symptom**: Analysis times out or runs out of memory  
**Cause**: Model too complex for comprehensive analysis  
**Solution**: 
- Use specific analysis types instead of "comprehensive"
- Disable performance benchmarking for large models
- Process models individually instead of batch
- Increase system memory or use sampling

#### Issue 2: Complexity metrics return zero
**Symptom**: Complexity calculations return zero or invalid values  
**Cause**: Model structure not properly extracted or missing components  
**Solution**:
- Verify GNN processing (step 3) completed successfully
- Check that model has variables and connections
- Use `--verbose` flag for detailed extraction logs

#### Issue 3: Framework comparison fails
**Symptom**: Cross-framework comparison reports errors  
**Cause**: Execution results (step 12) not available or incomplete  
**Solution**:
- Ensure execution step (12) completed successfully
- Verify framework outputs exist in execution results
- Check execution results format matches expected structure

---

## Version History

### Current Version: 1.0.0

**Features**:
- Statistical analysis
- Complexity metrics calculation
- Performance benchmarking
- Model comparison
- Framework output analysis

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced visualization of analysis results
- **Future**: Real-time analysis dashboard

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Execute Module](../execute/AGENTS.md)
- [Analysis Module](../analysis/README.md)

### External Resources
- [NetworkX Documentation](https://networkx.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://scipy.org/)

---

**Last Updated**: 2025-12-30
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
