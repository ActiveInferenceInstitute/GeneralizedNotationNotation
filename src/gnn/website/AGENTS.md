# Website Module - Agent Scaffolding

## Module Overview

**Purpose**: Static HTML website generation from pipeline artifacts and results

**Pipeline Step**: Step 20: Website generation (src/gnn/20_website.py)

**Category**: Documentation / Website Generation

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-04

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

#### `process_website(target_dir: Path, output_dir: Path, verbose: bool = False, pipeline_output_root: Path | None = None, **kwargs) -> bool`
**Description**: Main website generation function called by orchestrator (src/gnn/20_website.py). Generates a multi-page static HTML website from pipeline artifacts.

**Parameters**:
- `target_dir` (Path): Directory containing pipeline artifacts
- `output_dir` (Path): Output directory for website files
- `verbose` (bool): Enable verbose logging (default: False)
- `pipeline_output_root` (Path | None): Root of numbered pipeline output dirs; defaults to `output_dir.parent`
- `**kwargs`: Additional website generation options (e.g. `website_html_filename` from the orchestrator, accepted and ignored)

**Returns**: `bool` - True if website generation succeeded, False otherwise

**Example**:
```python
from gnn.website import process_website
from pathlib import Path

success = process_website(
    target_dir=Path("output"),
    output_dir=Path("output/20_website_output"),
    verbose=True,
)
```

#### `generate_html_report(content: str, output_file: Path) -> bool`
**Description**: Generate an HTML report from content and write it to `output_file`.

**Returns**: `bool` - True if writing succeeded, False otherwise

#### `embed_image(image_path: Path, output_file: Path) -> bool`
**Description**: Embed image in HTML output file.

**Returns**: `bool` - True if embedding succeeded, False otherwise

#### `embed_markdown_file(md_path: Path, output_file: Path) -> bool`
**Description**: Embed markdown file in HTML output.

**Returns**: `bool` - True if embedding succeeded, False otherwise

Additional exports (see `__init__.py`): `WebsiteGenerator`, `WebsiteRenderer`, `generate_website`, `embed_text_file`, `embed_json_file`, `embed_html_file`, `get_module_info`, `get_supported_file_types`, `validate_website_config`, `render_dashboard` (re-exported from `dashboard.py`), `collect_website_data`, `get_pipeline_steps`, `PIPELINE_STEPS`, `StepInfo`, `inspect_website`, `list_website_pages`.

#### `collect_website_data(pipeline_output_root, input_dir, assets_dir, *, output_dir=None, user_data=None, mcp_tools_provider=None) -> dict`
**Description**: Pure aggregation of every artifact the pages render (GNN files, step statuses, analysis JSON, execution summary, visualization assets, reports, MCP inventory). `mcp_tools_provider` injects the MCP tool list; the default performs a best-effort live registry read. Callers can build the data once and render pages from it.

#### `get_pipeline_steps() -> tuple[StepInfo, ...]`
**Description**: Returns the immutable 25-step catalogue (`StepInfo(number, name, description)` with a `script_name` display property) used to render the dashboard and pipeline pages.

#### `inspect_website(directory) -> dict` / `list_website_pages(directory) -> dict`
**Description**: Pure filesystem queries over a generated site (page inventory, sizes, key-page completeness; per-page size/mtime listing). `website.inspection.KEY_PAGES` lists the seven canonical pages. These are the shared implementation behind the `get_website_status` and `list_generated_website_pages` MCP tools.

---

## Dependencies

The website module is stdlib-only (`logging`, `pathlib`, `json`, `shutil`, `datetime`, `html`); no optional pip extra is required to import or run it. The Jinja2/Markdown/Bleach templating stack is not used — pages are built with inline CSS/HTML. (The orchestrator `src/gnn/20_website.py` pulls in `utils.pipeline_template`, a core utility.)

---

## Usage Example

```python
from gnn.website import embed_image

success = embed_image(
    image_path="visualizations/network.png", output_file="website/index.html"
)
```

---

## Output Specification

### Output Products
`generate_website` (in `generator.py`) writes seven pages plus a results manifest:
- `index.html` - Pipeline dashboard with step cards
- `pipeline.html` - Full 25-step pipeline status table
- `gnn_files.html` - GNN source file browser
- `analysis.html` - Analysis and complexity metrics
- `visualization.html` - Gallery of generated visualizations
- `reports.html` - JSON/text report viewer
- `mcp.html` - MCP tools registry across all modules
- `website_results.json` - generation manifest with `success`, `pages_created`, `pages` (written filenames), `errors`, `warnings`, `generated_at` (written by `process_website`)

`assets/` is created under the output dir; `static/` is copied only if a `static/` directory ships beside the module.

### Output Directory Structure
```
output/20_website_output/
├── index.html
├── pipeline.html
├── gnn_files.html
├── analysis.html
├── visualization.html
├── reports.html
├── mcp.html
├── website_results.json
└── assets/
```

---

## Performance Characteristics

Generation is fast (seconds) for typical pipeline output; no published benchmarks. Measure on demand if needed.

---
## Error Handling

### Page Resilience
Each of the seven pages is rendered and written independently (atomic temp-file + rename per page). A failure on one page records `Failed to render/write <page>` in `errors` and leaves the remaining pages intact; `success` in the result dict (and the `process_website` bool) is `True` only when no errors occurred. Values coming from pipeline data (GNN sources, analysis JSON, report content, MCP tool fields) are HTML-escaped on every page.

### Recovery Strategies
- **Template Recovery**: Use default templates
- **Content Simplification**: Simplify content processing
- **Asset Skip**: Skip problematic assets
- **Error Documentation**: Generate error reports

---

## Integration Points

### Orchestrated By
- **Script**: `src/gnn/20_website.py` (Step 20)
- **Function**: `process_website()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests/website/test_website_overall.py`, `test_website_public_api.py`, `test_website_dashboard.py` - Website tests

### Data Flow
```
Pipeline Artifacts → Content Extraction → Template Processing → Asset Embedding → Website Generation
```

---

## Testing

### Test Files
- `tests/website/test_website_overall.py` - Module-level tests
- `tests/website/test_website_public_api.py` - Public API surface tests
- `tests/website/test_website_dashboard.py` - Dashboard tests
- `tests/website/test_website_generator_units.py` - Catalogue, data collection, escaping, page-resilience, manifest tests
- `tests/website/test_website_inspection.py` - `inspect_website` / `list_website_pages` tests
### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/website/ \
    --cov=src/gnn/website --cov-report=term-missing
```
### Key Test Scenarios
1. Website generation from pipeline artifacts
2. HTML report creation and formatting
3. Asset embedding and management
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `process_website` - Generate a website from a target directory
- `build_website_from_pipeline_output` - Discover numbered pipeline artifacts and build the site
- `get_website_status` - Inspect completeness of an existing generated site
- `list_generated_website_pages` - List generated HTML pages and metadata
- `get_website_module_info` - Return website features and the live MCP inventory

The module-info inventory and `register_tools()` use these same five names.

### MCP File Location
- `src/gnn/website/mcp.py` - MCP tool registrations

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

Module `__version__` is `1.7.0` (`__init__.py`); the pipeline/repo release is `3.2.0`. No formal changelog is maintained in this file.

---
## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Website Module](../website/README.md)

### External Resources
- [HTML5 Specification](https://html.spec.whatwg.org/)

---

**Last Updated**: 2026-09-04
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern (delegates to `website.processor.process_website` → `renderer.process_website` → `generator.generate_website`)

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
