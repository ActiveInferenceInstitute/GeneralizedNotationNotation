# Specification: Website

## Design Requirements

The `src/website/` module generates static HTML websites from pipeline artifacts (Step 20).

## Interface Mapping

- `20_website.py`: Thin orchestrator binding `website.processor.process_website()`
- `processor.py`: Thin shim delegating to `generator.py` and `renderer.py`
- `generator.py`: Core HTML/CSS generation engine producing multi-page sites
- `renderer.py`: Template rendering with Jinja2 support
- `dashboard.py`: Interactive dashboard generation with visualization embedding
- `mcp.py`: MCP tool registration for website generation operations

## Functional Requirements

- **Static Site Generation**: Produce self-contained HTML websites from pipeline output artifacts
- **Template Rendering**: Jinja2-based template system for consistent page layouts
- **Visualization Embedding**: Embed graphs, charts, and interactive plots from Steps 8–9
- **Dashboard Generation**: Create interactive dashboards summarizing pipeline execution results
- **Cross-Referencing**: Link between model pages, execution results, and analysis reports

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `WebsiteGenerator` | Class | Core site generation engine |
| `WebsiteRenderer` | Class | Template rendering and HTML output |
| `process_website()` | Function | Top-level entry point called by orchestrator |
| `dashboard.py` | Module | Interactive dashboard generation |

## Standards

- Generated sites are fully self-contained (no external CDN dependencies at runtime)
- HTML5 semantic markup with responsive CSS layouts
- Graceful degradation when Jinja2 is unavailable (minimal HTML fallback)
- All generated pages include navigation, breadcrumbs, and cross-references
