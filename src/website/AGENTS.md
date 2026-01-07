# Website Module - Agent Scaffolding

## Module Overview

**Purpose**: Static HTML website generation from pipeline artifacts and results

**Pipeline Step**: Step 20: Website generation (20_website.py)

**Category**: Documentation / Website Generation

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

---

## Core Functionality

### Primary Responsibilities
1. Generate static HTML websites from pipeline results
2. Create interactive documentation and reports
3. Organize and present pipeline artifacts
4. Generate cross-linked documentation
5. Create publication-ready websites

### Key Capabilities
- Static website generation from pipeline artifacts
- Interactive documentation and reports
- Cross-linked content organization
- Publication-ready HTML output
- Asset management and optimization

---

## API Reference

### Public Functions

#### `process_website(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main website generation function called by orchestrator (20_website.py). Generates static HTML website from pipeline artifacts.

**Parameters**:
- `target_dir` (Path): Directory containing pipeline artifacts
- `output_dir` (Path): Output directory for website files
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `website_html_filename` (str, optional): Output HTML filename (default: "gnn_pipeline_summary_website.html")
- `include_visualizations` (bool, optional): Include visualization pages (default: True)
- `include_reports` (bool, optional): Include report pages (default: True)
- `include_analysis` (bool, optional): Include analysis pages (default: True)
- `**kwargs`: Additional website generation options

**Returns**: `bool` - True if website generation succeeded, False otherwise

**Example**:
```python
from website import process_website
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_website(
    target_dir=Path("output"),
    output_dir=Path("output/20_website_output"),
    logger=logger,
    verbose=True,
    website_html_filename="custom_summary.html"
)
```

#### `generate_html_report(content: Union[str, Dict[str, Any]], title: str = "Report") -> str`
**Description**: Generate HTML report from content (markdown or structured data).

**Parameters**:
- `content` (Union[str, Dict[str, Any]]): Content to convert to HTML (markdown string or structured dict)
- `title` (str): Report title (default: "Report")

**Returns**: `str` - HTML string with formatted report

#### `embed_image(image_path: Path, output_file: Path, alt_text: str = "") -> bool`
**Description**: Embed image in HTML output file.

**Parameters**:
- `image_path` (Path): Path to image file
- `output_file` (Path): Output HTML file to embed image in
- `alt_text` (str): Alternative text for image (default: "")

**Returns**: `bool` - True if embedding succeeded, False otherwise

**Returns**: `True` if embedding succeeded

#### `embed_markdown_file(md_path, output_file) -> bool`
**Description**: Embed markdown file in HTML output

**Parameters**:
- `md_path`: Path to markdown file
- `output_file`: Output HTML file

**Returns**: `True` if embedding succeeded

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation
- `jinja2` - HTML templating

### Optional Dependencies
- `markdown` - Markdown to HTML conversion
- `bleach` - HTML sanitization

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Website Settings
```python
WEBSITE_CONFIG = {
    'template': 'default',
    'theme': 'modern',
    'include_navigation': True,
    'generate_sitemap': True,
    'optimize_assets': True
}
```

### Content Settings
```python
CONTENT_CONFIG = {
    'include_pipeline_summary': True,
    'include_visualizations': True,
    'include_reports': True,
    'include_raw_data': False
}
```

---

## Usage Examples

### Basic Website Generation
```python
from website import process_website

success = process_website(
    target_dir="output/",
    output_dir="output/20_website_output"
)
```

### HTML Report Generation
```python
from website import generate_html_report

html_content = generate_html_report(markdown_content)
with open("report.html", "w") as f:
    f.write(html_content)
```

### Asset Embedding
```python
from website import embed_image

success = embed_image(
    image_path="visualizations/network.png",
    output_file="website/index.html"
)
```

---

## Output Specification

### Output Products
- `index.html` - Main website page
- `*.html` - Individual report pages
- `assets/` - Static assets and resources
- `sitemap.xml` - Website sitemap
- `website_summary.json` - Website generation summary

### Output Directory Structure
```
output/20_website_output/
├── index.html
├── pipeline_summary.html
├── visualizations.html
├── reports.html
├── assets/
│   ├── css/
│   ├── js/
│   └── images/
└── website_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-5 seconds
- **Memory**: ~50-100MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Basic Generation**: 1-2 seconds
- **Full Website**: 3-5 seconds
- **Asset Optimization**: 1-3 seconds
- **Content Processing**: 2-4 seconds

---

## Error Handling

### Website Errors
1. **Template Errors**: Template rendering failures
2. **Content Errors**: Content processing failures
3. **Asset Errors**: Asset embedding failures
4. **File I/O**: File system operation failures

### Recovery Strategies
- **Template Fallback**: Use default templates
- **Content Simplification**: Simplify content processing
- **Asset Skip**: Skip problematic assets
- **Error Documentation**: Generate error reports

---

## Integration Points

### Orchestrated By
- **Script**: `20_website.py` (Step 20)
- **Function**: `process_website()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_website_*` - Website tests

### Data Flow
```
Pipeline Artifacts → Content Extraction → Template Processing → Asset Embedding → Website Generation
```

---

## Testing

### Test Files
- `src/tests/test_website_integration.py` - Integration tests
- `src/tests/test_website_generation.py` - Generation tests

### Test Coverage
- **Current**: 79%
- **Target**: 85%+

### Key Test Scenarios
1. Website generation from pipeline artifacts
2. HTML report creation and formatting
3. Asset embedding and management
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `website.generate` - Generate website from artifacts
- `website.create_report` - Create HTML reports
- `website.embed_assets` - Embed assets in HTML
- `website.validate_content` - Validate website content

### Tool Endpoints
```python
@mcp_tool("website.generate")
def generate_website_tool(artifacts_dir, output_dir):
    """Generate website from pipeline artifacts"""
    # Implementation
```

### MCP File Location
- `src/website/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Website generation fails
**Symptom**: HTML files not generated or incomplete  
**Cause**: Missing pipeline artifacts or template issues  
**Solution**: 
- Verify previous pipeline steps completed successfully
- Check that required artifacts exist in output directories
- Use `--verbose` flag for detailed generation logs
- Review website template structure

#### Issue 2: Embedded content missing
**Symptom**: Website generated but images or markdown not embedded  
**Cause**: File paths incorrect or files missing  
**Solution**:
- Verify all referenced files exist
- Check file paths are relative to website output directory
- Ensure images and markdown files are accessible
- Review embedding function logs

---

## Version History

### Current Version: 1.0.0

**Features**:
- Static HTML website generation
- Interactive documentation
- Cross-linked content
- Asset management

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced interactivity
- **Future**: Dynamic content generation

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Website Module](../website/README.md)

### External Resources
- [HTML5 Specification](https://html.spec.whatwg.org/)

---

**Last Updated**: 2025-12-30
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern