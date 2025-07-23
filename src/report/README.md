# Report Generation Module

This module provides comprehensive analysis and reporting capabilities for the GNN pipeline. It generates unified reports from all pipeline outputs including HTML, Markdown, and JSON formats.

## Module Structure

```
src/report/
├── __init__.py          # Module exports and API
├── generator.py         # Main report generation orchestration
├── analyzer.py          # Data collection and analysis
├── formatters.py        # HTML and Markdown report formatting
└── README.md           # This documentation
```

## API Reference

### Main Functions

#### `generate_comprehensive_report(pipeline_output_dir, report_output_dir, logger)`
Main function that orchestrates the entire report generation process.

**Parameters:**
- `pipeline_output_dir` (Path): Directory containing all pipeline outputs
- `report_output_dir` (Path): Directory to save generated reports
- `logger` (Logger): Logger instance for operation tracking

**Returns:** `bool` - True if successful, False otherwise

**Outputs:**
- `comprehensive_analysis_report.html` - Interactive HTML report
- `comprehensive_analysis_report.md` - Markdown report
- `report_summary.json` - JSON data summary

### Data Analysis

#### `collect_pipeline_data(pipeline_output_dir, logger)`
Collects and analyzes data from all pipeline step directories.

**Returns:** `Dict[str, Any]` - Structured pipeline data

#### `analyze_step_directory(step_path, step_name, logger)`
Analyzes a specific step directory and extracts metrics.

**Returns:** `Dict[str, Any]` - Step analysis data including file counts, sizes, and types

### Report Formatting

#### `generate_html_report(pipeline_data, logger)`
Generates an interactive HTML report with styling and navigation.

**Returns:** `str` - Complete HTML content

#### `generate_markdown_report(pipeline_data, logger)`
Generates a clean Markdown report for documentation.

**Returns:** `str` - Complete Markdown content

## Usage Examples

### Basic Usage
```python
from report.generator import generate_comprehensive_report
from pathlib import Path
import logging

# Generate comprehensive report
success = generate_comprehensive_report(
    pipeline_output_dir=Path("output"),
    report_output_dir=Path("output/report_processing_step"),
    logger=logging.getLogger("report")
)
```

### Direct Analysis
```python
from report.analyzer import collect_pipeline_data, analyze_step_directory

# Collect all pipeline data
pipeline_data = collect_pipeline_data(Path("output"), logger)

# Analyze specific step
step_data = analyze_step_directory(Path("output/visualization"), "visualization", logger)
```

### Custom Formatting
```python
from report.formatters import generate_html_report, generate_markdown_report

# Generate custom reports
html_content = generate_html_report(pipeline_data, logger)
markdown_content = generate_markdown_report(pipeline_data, logger)
```

## Pipeline Integration

This module is used by the `14_report.py` pipeline step, which follows the standardized pipeline template pattern:

```python
from utils.pipeline_template import create_standardized_pipeline_script
from report.generator import generate_comprehensive_report

def process_report_generation(target_dir, output_dir, logger, **kwargs):
    # Standardized processing function
    return generate_comprehensive_report(output_dir, report_output_dir, logger)

run_script = create_standardized_pipeline_script(
    "14_report.py",
    process_report_generation,
    "Comprehensive analysis report generation"
)
```

## Supported Pipeline Steps

The report generator analyzes the following pipeline step directories:

- `setup_artifacts/` - Environment setup results
- `gnn_processing_step/` - GNN discovery and parsing
- `test_reports/` - Test execution results
- `type_check/` - Type checking reports
- `gnn_exports/` - Multi-format exports
- `visualization/` - Generated visualizations
- `mcp_processing_step/` - MCP integration reports
- `ontology_processing/` - Ontology analysis
- `gnn_rendered_simulators/` - Generated code
- `execution_results/` - Simulation results
- `llm_processing_step/` - LLM analysis
- `audio_processing_step/` - Audio generation
- `website/` - HTML documentation
- `report_processing_step/` - Previous report outputs

## Report Features

### HTML Report
- Interactive styling with CSS
- Step-by-step analysis cards
- File type breakdowns
- Size and modification time tracking
- Pipeline execution summary integration

### Markdown Report
- Clean, documentation-friendly format
- Structured sections and subsections
- Easy integration with documentation systems
- Version control friendly

### JSON Summary
- Machine-readable data format
- Complete pipeline metrics
- Integration with other analysis tools
- API consumption ready

## Error Handling

The module includes comprehensive error handling:

- Graceful degradation when step directories are missing
- Detailed logging of analysis failures
- Fallback behavior for corrupted data
- Validation of output file generation

## Performance Considerations

- Efficient file system traversal using `pathlib.rglob()`
- Minimal memory usage through streaming analysis
- Optimized file size calculations
- Cached timestamp conversions

## Dependencies

- Standard library: `pathlib`, `json`, `datetime`, `logging`
- No external dependencies required
- Compatible with Python 3.8+ 