# Website Module - Agent Scaffolding

## Module Overview

**Purpose**: Static HTML website generation from pipeline artifacts with embedded visualizations and comprehensive documentation

**Pipeline Step**: Step 20: Website generation (20_website.py)

**Category**: Documentation / Website Generation

---

## Core Functionality

### Primary Responsibilities
1. Generate static HTML websites from pipeline results
2. Embed visualizations, charts, and analysis results
3. Create comprehensive documentation sites
4. Support multiple output formats and themes
5. Enable interactive exploration of pipeline artifacts

### Key Capabilities
- Static HTML website generation
- Multi-format content embedding (images, markdown, JSON, HTML)
- Responsive design and mobile compatibility
- Interactive visualizations and charts
- Cross-reference linking between pipeline steps
- Customizable themes and styling

---

## API Reference

### Public Functions

#### `process_website(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main website generation function from pipeline artifacts

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for website
- `logger` (Logger): Logger instance for progress reporting
- `website_theme` (str): Website theme ("default", "modern", "minimal")
- `include_interactive` (bool): Include interactive elements
- `**kwargs`: Additional website-specific options

**Returns**: `True` if website generation succeeded

#### `generate_website(artifacts_dir, output_dir, **kwargs) -> Dict[str, Any]`
**Description**: Generate complete website from pipeline artifacts

**Parameters**:
- `artifacts_dir` (Path): Directory containing pipeline artifacts
- `output_dir` (Path): Output directory for website
- `theme` (str): Website theme and styling
- `**kwargs`: Additional generation options

**Returns**: Dictionary with website generation results

#### `WebsiteGenerator` - Website Generation Class
**Description**: Main class for website generation and management

**Key Methods**:
- `generate()` - Generate complete website
- `add_page()` - Add custom page to website
- `embed_content()` - Embed external content
- `set_theme()` - Apply website theme

---

## Website Features

### Content Embedding
**Supported Formats**:
- **Images**: PNG, JPG, JPEG, GIF, SVG
- **Text**: Markdown, plain text, RST
- **Data**: JSON, YAML, CSV
- **HTML**: Custom HTML content
- **Visualizations**: Charts, graphs, interactive plots

### Navigation Structure
**Standard Pages**:
- **Index**: Main landing page with overview
- **Pipeline**: Step-by-step pipeline documentation
- **Models**: GNN model specifications and details
- **Analysis**: Statistical analysis and results
- **Visualizations**: Charts, graphs, and interactive displays
- **Code**: Generated simulation code and examples
- **Documentation**: Comprehensive API reference

### Interactive Features
**Capabilities**:
- Collapsible sections and accordions
- Tabbed interfaces for different views
- Search functionality across content
- Responsive design for mobile devices
- Dark/light theme switching

### Theme System
**Available Themes**:
- **Default**: Clean, professional layout
- **Modern**: Contemporary design with animations
- **Minimal**: Simple, distraction-free interface
- **Custom**: User-defined themes and styling

---

## Dependencies

### Required Dependencies
- `pathlib` - File system operations and path handling
- `os` - System operations and file management
- `shutil` - File copying and directory operations
- `json` - Data serialization and configuration

### Optional Dependencies
- `jinja2` - HTML template rendering (fallback: basic HTML)
- `markdown` - Markdown processing (fallback: plain text)
- `beautifulsoup4` - HTML parsing and manipulation (fallback: regex)
- `pillow` - Image processing and optimization (fallback: copy only)

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `WEBSITE_THEME` - Default website theme ("default", "modern", "minimal")
- `WEBSITE_TITLE` - Website title and branding
- `WEBSITE_AUTHOR` - Website author information
- `WEBSITE_INCLUDE_INTERACTIVE` - Include interactive elements (default: True)

### Configuration Files
- `website_config.yaml` - Website generation settings and themes

### Default Settings
```python
DEFAULT_WEBSITE_SETTINGS = {
    'theme': 'default',
    'title': 'GNN Pipeline Results',
    'author': 'Active Inference Institute',
    'include_interactive': True,
    'embed_visualizations': True,
    'embed_code_examples': True,
    'responsive_design': True,
    'search_enabled': True,
    'theme_switching': True,
    'sections': {
        'overview': True,
        'pipeline': True,
        'models': True,
        'analysis': True,
        'visualizations': True,
        'code': True,
        'api': True
    }
}
```

---

## Usage Examples

### Basic Website Generation
```python
from website.generator import generate_website

result = generate_website(
    artifacts_dir=Path("output/"),
    output_dir=Path("output/20_website_output"),
    theme="modern"
)
print(f"Website generated: {result['success']}")
```

### Website Generation Class
```python
from website.generator import WebsiteGenerator

generator = WebsiteGenerator(
    artifacts_dir=Path("output/"),
    output_dir=Path("output/20_website_output"),
    theme="default"
)

success = generator.generate()
print(f"Pages generated: {len(generator.pages)}")
```

### Content Embedding
```python
from website.renderer import embed_markdown_file

html_content = embed_markdown_file(
    Path("output/16_analysis_output/model_analysis.md"),
    title="Statistical Analysis"
)
```

---

## Output Specification

### Output Products
- `index.html` - Main website landing page
- `pipeline.html` - Pipeline step documentation
- `models.html` - GNN model specifications
- `analysis.html` - Statistical analysis results
- `visualizations.html` - Charts and interactive displays
- `code.html` - Generated simulation code
- `api.html` - API documentation and reference
- `assets/` - CSS, JavaScript, images, and resources

### Output Directory Structure
```
output/20_website_output/
├── index.html
├── pipeline.html
├── models.html
├── analysis.html
├── visualizations.html
├── code.html
├── api.html
├── assets/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── fonts/
└── website_generation_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~10-30 seconds (website generation)
- **Memory**: ~50-100MB for comprehensive sites
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Path**: ~5-10s for basic websites
- **Slow Path**: ~30-60s for comprehensive sites with many assets
- **Memory**: ~20-50MB for typical sites, ~100MB+ for large sites

---

## Error Handling

### Graceful Degradation
- **No template engine**: Fallback to basic HTML templates
- **Missing assets**: Generate placeholder content with warnings
- **Large sites**: Streaming generation with progress updates

### Error Categories
1. **Template Errors**: Invalid or missing website templates
2. **Asset Errors**: Missing or corrupted assets and resources
3. **Content Errors**: Invalid or malformed content for embedding
4. **Resource Errors**: Memory or disk space exhaustion

---

## Integration Points

### Orchestrated By
- **Script**: `20_website.py` (Step 20)
- **Function**: `process_website()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_website_unit.py` - Website generation tests
- `main.py` - Pipeline orchestration

### Data Flow
```
Pipeline Artifacts → Content Aggregation → Template Processing → Asset Embedding → Static Website
```

---

## Testing

### Test Files
- `src/tests/test_website_unit.py` - Unit tests
- `src/tests/test_website_integration.py` - Integration tests
- `src/tests/test_website_embedding.py` - Content embedding tests

### Test Coverage
- **Current**: 79%
- **Target**: 90%+

### Key Test Scenarios
1. Website generation across different themes
2. Content embedding for various file types
3. Large site generation and performance
4. Error handling with missing assets
5. Mobile responsiveness and accessibility

---

## MCP Integration

### Tools Registered
- `website_generate` - Generate websites from pipeline data
- `website_embed` - Embed content in websites
- `website_theme` - Apply themes to websites

### Tool Endpoints
```python
@mcp_tool("website_generate")
def generate_website_tool(artifacts_dir, theme="default", interactive=True):
    """Generate website from pipeline artifacts"""
    # Implementation
```

---

**Last Updated**: September 30, 2025
**Status**: ✅ Production Ready
