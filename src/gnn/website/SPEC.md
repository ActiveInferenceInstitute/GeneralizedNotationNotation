# Specification: Website

## Design Requirements

The `src/gnn/website/` module generates static HTML websites from pipeline artifacts (Step 20).

## Interface Mapping

- `20_website.py`: Thin orchestrator binding `website.processor.process_website()`
- `processor.py`: Thin facade re-exporting `renderer.process_website`
- `collection.py`: Pipeline-artifact collectors behind `collect_website_data` (GNN files, step statuses, analysis, visualization assets, reports, MCP page data)
- `generator.py`: Core HTML/CSS generation engine producing the 7-page site (rich pipeline-summary data folded into the generated index page)
- `renderer.py`: `process_website`, embedding helpers, and module info
- `mcp.py`: MCP tool registration for website generation operations

## Functional Requirements

- **Static Site Generation**: Produce self-contained HTML websites from pipeline output artifacts
- **Inline Templating**: Pages are built with inline CSS/HTML (no Jinja2 dependency)
- **Visualization Embedding**: Embed graphs and images from Steps 8–9
- **Cross-Referencing**: Link between model pages, execution results, and analysis reports

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `WebsiteGenerator` | Class | Core site generation engine |
| `WebsiteRenderer` | Class | Template rendering and HTML output |
| `process_website()` | Function | Top-level entry point called by orchestrator |

## Standards

- Generated sites are self-contained (inline CSS/JS; no external CDN)
- HTML5 semantic markup with responsive CSS layouts
- No Jinja2/Markdown/Bleach dependency — stdlib only
- All generated pages include navigation, breadcrumbs, and cross-references
