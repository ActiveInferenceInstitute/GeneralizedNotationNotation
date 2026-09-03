# Specification: Website

## Design Requirements

The `src/website/` module generates static HTML websites from pipeline artifacts (Step 20).

## Interface Mapping

- `20_website.py`: Thin orchestrator binding `website.processor.process_website()`
- `processor.py`: Thin facade re-exporting `renderer.process_website`
- `generator.py`: Core HTML/CSS generation engine producing the 7-page site
- `renderer.py`: `process_website`, embedding helpers, and module info
- `dashboard.py`: Standalone interactive dashboard (`render_dashboard`); loads Mermaid from CDN
- `mcp.py`: MCP tool registration for website generation operations

## Functional Requirements

- **Static Site Generation**: Produce self-contained HTML websites from pipeline output artifacts
- **Inline Templating**: Pages are built with inline CSS/HTML (no Jinja2 dependency)
- **Visualization Embedding**: Embed graphs and images from Steps 8–9
- **Dashboard Generation**: `dashboard.py` renders an interactive dashboard (CDN-loaded Mermaid)
- **Cross-Referencing**: Link between model pages, execution results, and analysis reports

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `WebsiteGenerator` | Class | Core site generation engine |
| `WebsiteRenderer` | Class | Template rendering and HTML output |
| `process_website()` | Function | Top-level entry point called by orchestrator |
| `dashboard.py` | Module | Interactive dashboard generation |

## Standards

- Generated sites are self-contained (inline CSS/JS; no external CDN) except the standalone `dashboard.py` page, which loads Mermaid from a CDN
- HTML5 semantic markup with responsive CSS layouts
- No Jinja2/Markdown/Bleach dependency — stdlib only
- All generated pages include navigation, breadcrumbs, and cross-references
