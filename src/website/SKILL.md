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

```python
from website import (
    WebsiteGenerator,
    WebsiteRenderer,
    generate_website,
    process_website,
    generate_html_report,
    embed_image,
    embed_markdown_file,
    embed_text_file,
    get_supported_file_types,
    validate_website_config,
)

# Process website step (used by pipeline)
process_website(target_dir, output_dir, verbose=True)

# Module-level convenience (generator.py): generate_website(logger, input_dir, output_dir, *, pipeline_output_root=None)
result = generate_website(logger, target_dir, output_dir)

# WebsiteGenerator builds the 7-page site from a website_data dict
gen = WebsiteGenerator()
gen.generate_website({"input_dir": str(target_dir), "output_dir": str(output_dir)})

# Generate an individual HTML report (content: str, output_file: Path) -> bool
generate_html_report(content, output_path)

# Embed content (each returns bool)
embed_image(image_path, output_file)
embed_markdown_file(md_path, output_file)

# Query supported file types
types = get_supported_file_types()
```

## Key Exports

- `WebsiteGenerator` / `WebsiteRenderer` — website generation classes
- `generate_website` / `process_website` — main generation functions
- `generate_html_report` — HTML report generation; takes content text and an
  output file path, returns True on success
- `embed_image`, `embed_markdown_file`, `embed_text_file`, `embed_json_file`, `embed_html_file`
- `validate_website_config` — configuration validation

## Output

- Static HTML site in `output/20_website_output/`
- Self-contained (no external dependencies at runtime)


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `build_website_from_pipeline_output`
- `get_website_module_info`
- `get_website_status`
- `list_generated_website_pages`
- `process_website`

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
