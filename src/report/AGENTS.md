# Report Module - Agent Scaffolding

## Module Overview

**Purpose**: Comprehensive analysis report generation with multiple format support and automated documentation creation

**Pipeline Step**: Step 23: Report generation (23_report.py)

**Category**: Documentation / Analysis Reporting

---

## Core Functionality

### Primary Responsibilities
1. Generate comprehensive analysis reports from pipeline results
2. Create multi-format documentation (HTML, Markdown, JSON, PDF)
3. Aggregate results from all pipeline steps into unified reports
4. Provide automated report generation and formatting
5. Enable customizable report templates and styling

### Key Capabilities
- Multi-format report generation (HTML, Markdown, JSON, PDF)
- Pipeline results aggregation and analysis
- Automated documentation creation
- Customizable report templates and styling
- Interactive HTML reports with visualizations
- Cross-reference linking between pipeline steps

---

## API Reference

### Public Functions

#### `process_report(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main report processing function for comprehensive analysis reporting

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for report results
- `logger` (Logger): Logger instance for progress reporting
- `report_format` (str): Report format ("comprehensive", "summary", "technical")
- `include_visualizations` (bool): Include visualizations in report
- `**kwargs`: Additional report-specific options

**Returns**: `True` if report generation succeeded

#### `generate_comprehensive_report(target_dir, output_dir, **kwargs) -> Dict[str, Any]`
**Description**: Generate comprehensive analysis report from pipeline results

**Parameters**:
- `target_dir` (Path): Input directory with pipeline results
- `output_dir` (Path): Output directory for reports
- `format` (str): Report format ("html", "markdown", "json")
- `**kwargs`: Additional formatting options

**Returns**: Dictionary with report generation results

#### `generate_html_report(data, output_path) -> bool`
**Description**: Generate interactive HTML report from analysis data

**Parameters**:
- `data` (Dict): Analysis data to include in report
- `output_path` (Path): Output path for HTML file

**Returns**: `True` if HTML report generation succeeded

---

## Report Types and Formats

### Comprehensive Report
**Purpose**: Complete pipeline analysis with all details
**Features**:
- Executive summary and key findings
- Detailed step-by-step analysis
- Performance metrics and benchmarks
- Error analysis and recommendations
- Cross-reference linking

### Summary Report
**Purpose**: High-level overview for quick review
**Features**:
- Key metrics and success indicators
- Performance highlights
- Critical findings and warnings
- Action items and next steps

### Technical Report
**Purpose**: Detailed technical documentation
**Features**:
- Implementation details and architecture
- Configuration and environment information
- Troubleshooting and debugging information
- API reference and usage examples

### Supported Formats

#### HTML Format
- Interactive web-based reports
- Embedded visualizations and charts
- Collapsible sections and navigation
- Responsive design for mobile devices

#### Markdown Format
- Human-readable structured documentation
- GitHub-compatible formatting
- Easy integration with documentation systems
- Print-friendly layout

#### JSON Format
- Machine-readable structured data
- API integration and automation
- Data analysis and processing
- Metadata and configuration export

#### PDF Format (via external tools)
- Professional document format
- Print-ready layout and styling
- Archival and sharing purposes
- Integration with document management systems

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation and file operations
- `json` - JSON data handling and serialization
- `typing` - Type annotations and validation

### Optional Dependencies
- `jinja2` - HTML template rendering (fallback: basic HTML)
- `markdown` - Markdown processing (fallback: text-based)
- `weasyprint` - PDF generation (fallback: HTML only)
- `plotly` - Interactive charts in HTML (fallback: static images)

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `REPORT_FORMAT` - Default report format ("html", "markdown", "json")
- `REPORT_TEMPLATE` - Custom report template path
- `REPORT_INCLUDE_VISUALS` - Include visualizations in reports (default: True)
- `REPORT_COMPRESSION` - Enable report compression for large outputs

### Configuration Files
- `report_config.yaml` - Report generation settings and templates

### Default Settings
```python
DEFAULT_REPORT_SETTINGS = {
    'format': 'html',
    'template': 'default',
    'include_visualizations': True,
    'include_raw_data': False,
    'compression': False,
    'max_file_size': '100MB',
    'sections': {
        'executive_summary': True,
        'detailed_analysis': True,
        'performance_metrics': True,
        'error_analysis': True,
        'recommendations': True,
        'appendices': False
    }
}
```

---

## Usage Examples

### Basic Report Generation
```python
from report.processor import process_report

success = process_report(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/23_report_output"),
    logger=logger,
    report_format="comprehensive"
)
```

### HTML Report Generation
```python
from report.generator import generate_html_report

data = {
    'pipeline_results': {...},
    'analysis_data': {...},
    'performance_metrics': {...}
}

success = generate_html_report(
    data,
    Path("output/23_report_output/comprehensive_report.html")
)
```

### Custom Report Configuration
```python
from report.processor import generate_comprehensive_report

report = generate_comprehensive_report(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/23_report_output"),
    format="html",
    include_visualizations=True,
    template="custom"
)
```

---

## Output Specification

### Output Products
- `comprehensive_report.html` - Interactive HTML report
- `comprehensive_report.md` - Markdown documentation
- `comprehensive_report.json` - Structured data export
- `report_generation_summary.json` - Generation metadata
- `report_assets/` - Images, charts, and supporting files

### Output Directory Structure
```
output/23_report_output/
├── comprehensive_report.html
├── comprehensive_report.md
├── comprehensive_report.json
├── report_generation_summary.json
└── report_assets/
    ├── charts/
    ├── images/
    └── data/
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~5-15 seconds (depending on data size)
- **Memory**: ~50-100MB for comprehensive reports
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Path**: ~2-5s for summary reports
- **Slow Path**: ~10-30s for comprehensive reports with visuals
- **Memory**: ~20-50MB for typical reports, ~100MB+ for large datasets

---

## Error Handling

### Graceful Degradation
- **No template engine**: Fallback to basic HTML templates
- **No visualization libraries**: Generate text-based reports
- **Large datasets**: Sampling and summary generation

### Error Categories
1. **Template Errors**: Invalid or missing report templates
2. **Format Errors**: Unsupported output format requests
3. **Data Errors**: Invalid or corrupted pipeline data
4. **Resource Errors**: Memory or disk space exhaustion

---

## Integration Points

### Orchestrated By
- **Script**: `23_report.py` (Step 23)
- **Function**: `process_report()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_report_integration.py` - Report generation tests
- `main.py` - Pipeline orchestration

### Data Flow
```
Pipeline Results → Report Aggregation → Data Analysis → Format Generation → Multi-format Output
```

---

## Testing

### Test Files
- `src/tests/test_report_integration.py` - Integration tests
- `src/tests/test_report_generation.py` - Generation tests
- `src/tests/test_report_formats.py` - Format tests

### Test Coverage
- **Current**: 81%
- **Target**: 90%+

### Key Test Scenarios
1. Report generation across all supported formats
2. Template rendering and customization
3. Large dataset handling and performance
4. Error handling with malformed data
5. Integration with pipeline results

---

## MCP Integration

### Tools Registered
- `report_generate` - Generate comprehensive reports
- `report_format` - Convert reports between formats
- `report_analyze` - Analyze existing reports

### Tool Endpoints
```python
@mcp_tool("report_generate")
def generate_report(pipeline_data, format="html", template="default"):
    """Generate comprehensive report from pipeline data"""
    # Implementation
```

---

**Last Updated: October 28, 2025
**Status**: ✅ Production Ready
