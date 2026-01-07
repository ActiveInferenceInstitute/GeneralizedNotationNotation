# Research Module - Agent Scaffolding

## Module Overview

**Purpose**: Rule-based expert system for analyzing GNN models and generating research hypotheses.

**Pipeline Step**: Step 19: Research tools (19_research.py)

**Category**: Research / Experimental Analysis

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2026-01-07

---

## Core Functionality

### Primary Responsibilities
1. Advanced research analysis and experimentation
2. Research methodology implementation and validation
3. Experimental feature development and testing
4. Research data collection and analysis
5. Publication and documentation support
6. Research collaboration tools

### Key Capabilities
- **Rule-Based Hypothesis Generation**: Uses static analysis heuristics to suggest model improvements.
- **Complexity Analysis**: Detects high-dimensional matrices (>10 dims) to suggest reduction techniques.
- **Structural Diagnostics**: Analyzes variable-to-connection ratios to identify sparse causal structures.
- **Automated Reporting**: Generates markdown reports justifying every hypothesis with discovered evidence.

---

## API Reference

### Public Functions

#### `process_research(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main research processing function called by orchestrator (19_research.py). Performs rule-based hypothesis generation and research analysis.

**Parameters**:
- `target_dir` (Path): Directory containing research data (GNN files)
- `output_dir` (Path): Output directory for research results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `analysis_type` (str, optional): Type of analysis ("comprehensive", "statistical", "experimental") (default: "comprehensive")
- `generate_hypotheses` (bool, optional): Generate research hypotheses (default: True)
- `**kwargs`: Additional research options

**Returns**: `bool` - True if research processing succeeded, False otherwise

**Example**:
```python
from research import process_research
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_research(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/19_research_output"),
    logger=logger,
    verbose=True,
    analysis_type="comprehensive"
)
```

#### `perform_research_analysis(data: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]`
**Description**: Perform advanced research analysis using rule-based expert system.

**Parameters**:
- `data` (Dict[str, Any]): Research data to analyze (parsed GNN models)
- `analysis_type` (str): Type of analysis ("comprehensive", "statistical", "experimental")

**Returns**: `Dict[str, Any]` - Research analysis results with:
- `hypotheses` (List[Dict]): Generated research hypotheses
- `complexity_analysis` (Dict): Complexity metrics and diagnostics
- `structural_diagnostics` (Dict): Structural analysis results
- `recommendations` (List[str]): Research recommendations

#### `generate_research_report(analysis_results: Dict[str, Any], output_format: str = "markdown") -> str`
**Description**: Generate research report from analysis results with evidence-based justifications.

**Parameters**:
- `analysis_results` (Dict[str, Any]): Results from research analysis
- `output_format` (str): Output format ("markdown", "html", "json")

**Returns**: `str` - Research report as formatted string with evidence and justifications

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations
- `pandas` - Data analysis
- `matplotlib` - Research visualization

### Optional Dependencies
- `scipy` - Advanced statistical analysis
- `scikit-learn` - Machine learning research tools
- `jupyter` - Interactive research notebooks

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Research Settings
```python
RESEARCH_CONFIG = {
    'analysis_types': ['statistical', 'experimental', 'comparative'],
    'output_formats': ['markdown', 'html', 'pdf'],
    'visualization_style': 'publication',
    'statistical_significance': 0.05,
    'include_methodology': True
}
```

---

## Usage Examples

### Basic Research Analysis
```python
from research.processor import process_research

success = process_research(
    target_dir="research_data/",
    output_dir="output/19_research_output",
    analysis_type="comprehensive"
)
```

### Advanced Research Analysis
```python
from research.analyzer import perform_research_analysis

results = perform_research_analysis(
    data=experimental_data,
    analysis_type="experimental"
)
```

### Research Report Generation
```python
from research.generator import generate_research_report

report = generate_research_report(
    analysis_results=results,
    output_format="markdown"
)
```

---

## Output Specification

### Output Products
- `research_analysis_report.md` - Comprehensive research report
- `research_data_analysis.json` - Detailed analysis results
- `research_visualizations/` - Research visualizations
- `research_summary.json` - Research summary

### Output Directory Structure
```
output/19_research_output/
├── research_analysis_report.md
├── research_data_analysis.json
├── research_visualizations/
│   ├── statistical_plots.png
│   └── experimental_results.png
└── research_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: Variable (depends on research complexity)
- **Memory**: ~50-200MB for complex analyses
- **Status**: ✅ Production Ready

### Expected Performance
- **Statistical Analysis**: 1-5 minutes
- **Experimental Analysis**: 5-30 minutes
- **Report Generation**: < 1 minute
- **Visualization**: 30 seconds - 2 minutes

---

## Error Handling

### Research Errors
1. **Data Quality Issues**: Invalid or insufficient research data
2. **Analysis Failures**: Statistical or computational errors
3. **Visualization Errors**: Plot generation failures
4. **Report Generation**: Documentation creation errors

### Recovery Strategies
- **Data Cleaning**: Automatic data quality improvement
- **Analysis Fallback**: Alternative analysis methods
- **Visualization Fallback**: Simplified visualizations
- **Report Recovery**: Error-aware report generation

---

## Integration Points

### Orchestrated By
- **Script**: `19_research.py` (Step 19)
- **Function**: `process_research()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- Research-specific applications
- `tests.test_research_*` - Research tests

### Data Flow
```
Research Data → Analysis → Visualization → Report Generation → Publication
```

---

## Testing

### Test Files
- `src/tests/test_research_integration.py` - Integration tests
- `src/tests/test_research_analysis.py` - Analysis tests

### Test Coverage
- **Current**: 70%
- **Target**: 80%+

### Key Test Scenarios
1. Research analysis with various data types
2. Report generation and formatting
3. Visualization creation
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `research.analyze_data` - Perform research analysis
- `research.generate_report` - Generate research reports
- `research.create_visualization` - Create research visualizations
- `research.validate_methodology` - Validate research methodology

### Tool Endpoints
```python
@mcp_tool("research.analyze_data")
def analyze_research_data_tool(data, analysis_type="comprehensive"):
    """Perform research analysis on data"""
    # Implementation
```

### MCP File Location
- `src/research/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Hypothesis generation produces no results
**Symptom**: Research analysis completes but no hypotheses generated  
**Cause**: Model structure doesn't match rule patterns or analysis incomplete  
**Solution**: 
- Verify GNN model has complete structure
- Check that model has variables and connections
- Use `--verbose` flag for detailed analysis logs
- Review rule-based analysis patterns

#### Issue 2: Research report generation fails
**Symptom**: Analysis succeeds but report generation errors  
**Cause**: Report template issues or output format problems  
**Solution**:
- Check output directory permissions
- Verify report format is supported
- Review report template structure
- Use default markdown format if issues persist

---

## Version History

### Current Version: 1.0.0

**Features**:
- Rule-based hypothesis generation
- Complexity analysis
- Structural diagnostics
- Automated reporting

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced hypothesis generation
- **Future**: Machine learning-based hypothesis generation

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Research Module](../research/README.md)

### External Resources
- [Active Inference Research](https://activeinference.org/research)

---

**Last Updated**: 2026-01-07
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern