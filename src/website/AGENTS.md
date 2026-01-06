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

#### `process_website(target_dir, output_dir, verbose=False, logger=None, **kwargs) -> bool`
**Description**: Main website generation function called by orchestrator (20_website.py)

**Parameters**:
- `target_dir` (Path): Directory containing pipeline artifacts
- `output_dir` (Path): Output directory for website
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Logger, optional): Logger instance (default: None)
- `website_html_filename` (str): Output HTML filename (default: "gnn_pipeline_summary_website.html")
- `**kwargs`: Additional website options

**Returns**: `True` if website generation succeeded

**Example**:
```python
from website import process_website

success = process_website(
    target_dir=Path("output"),
    output_dir=Path("output/20_website_output"),
    verbose=True,
    website_html_filename="custom_summary.html"
)
```

#### `generate_html_report(content) -> str`
**Description**: Generate HTML report from content

**Parameters**:
- `content`: Content to convert to HTML

**Returns**: HTML string

#### `embed_image(image_path, output_file) -> bool`
**Description**: Embed image in HTML output

**Parameters**:
- `image_path`: Path to image file
- `output_file`: Output HTML file

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

---