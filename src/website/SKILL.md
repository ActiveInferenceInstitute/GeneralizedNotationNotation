---
name: gnn-website-generation
description: GNN static HTML website generation from pipeline artifacts. Use when generating browsable documentation websites, creating HTML galleries of model visualizations, or publishing pipeline results as a static site.
---

# GNN Website Generation (Step 20)

## Purpose

Generates a static HTML website from pipeline artifacts including model visualizations, export data, analysis reports, and documentation. Creates a browsable, self-contained web experience.

## Key Commands

```bash
# Generate website
python src/20_website.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 20 --verbose
```

## API

```python
from website import (
    WebsiteGenerator, WebsiteRenderer,
    generate_website, process_website,
    generate_html_report, get_module_info,
    embed_image, embed_markdown_file, embed_text_file,
    get_supported_file_types, validate_website_config
)

# Process website step (used by pipeline)
process_website(target_dir, output_dir, verbose=True)

# Use the WebsiteGenerator class
gen = WebsiteGenerator()
generate_website(artifacts_dir, output_dir)

# Generate individual HTML report
generate_html_report(data, output_path)

# Embed content
embed_image(image_path, html_output)
embed_markdown_file(md_path, html_output)

# Query supported file types
types = get_supported_file_types()
```

## Key Exports

- `WebsiteGenerator` / `WebsiteRenderer` — website generation classes
- `generate_website` / `process_website` — main generation functions
- `generate_html_report` — HTML report generation
- `embed_image`, `embed_markdown_file`, `embed_text_file`, `embed_json_file`, `embed_html_file`
- `validate_website_config` — configuration validation

## Output

- Static HTML site in `output/20_website_output/`
- Self-contained (no external dependencies at runtime)


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_website`
- `build_from_pipeline_output`
- `get_website_status`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
