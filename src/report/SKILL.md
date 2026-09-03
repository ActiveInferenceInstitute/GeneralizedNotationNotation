---
name: gnn-report-generation
description: GNN comprehensive analysis report generation. Use when creating summary reports from pipeline outputs, generating markdown or HTML reports, or producing executive summaries of model processing results.
---

# GNN Report Generation (Step 23)

## Purpose

Generates comprehensive analysis reports from all upstream pipeline outputs including parsing results, validation reports, simulation outcomes, statistical analyses, and visualizations.

## Key Commands

```bash
# Generate reports
python src/23_report.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 23 --verbose
```

## API

```python
from report import (
    process_report,
    generate_comprehensive_report,
    analyze_gnn_file,
    generate_html_report,
    generate_markdown_report,
    generate_report,
    ReportGenerator,
    ReportFormatter,
    get_module_info,
    get_supported_formats,
    validate_report,
)

# Process report step (used by pipeline)
process_report(target_dir, output_dir, verbose=True)

# Generate comprehensive report from pipeline outputs (generator variant)
ok = generate_comprehensive_report(
    pipeline_output_dir, report_output_dir, logger,
    report_formats=["html", "markdown", "json"],
)

# File-level GNN analysis (processor variant)
report = generate_comprehensive_report(target_dir, output_dir, format="html")

# Analyze a single GNN file
info = analyze_gnn_file(Path("input/gnn_files/model.md"))

# Render HTML/Markdown strings
html = generate_html_report(report_data)
md = generate_markdown_report(report_data)

# Validate a report
is_valid = validate_report(report_data)

# Query supported formats
formats = get_supported_formats()
```

## Key Exports

- `process_report` — main pipeline processing function
- `generate_comprehensive_report` / `generate_report` — report generation
- `ReportGenerator` / `ReportFormatter` — report classes
- `generate_html_report` / `generate_markdown_report` — format-specific
- `analyze_gnn_file` — file-level analysis for reports
- `validate_report` / `get_supported_formats` — utilities

## Output

- `output/23_report_output/comprehensive_analysis_report.html|.md` - analysis reports
- `report_summary.json`, `report_generation_summary.json`, `report_processing_summary.json` - machine-readable summaries


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `generate_report`
- `get_report_module_info`
- `list_report_formats`
- `process_report`
- `read_report`

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
