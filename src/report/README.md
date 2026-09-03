# Report Module

This module (Pipeline Step 23) consolidates pipeline outputs — per-step artifacts, logs, and metrics — into comprehensive HTML, Markdown, and JSON analysis reports, with pipeline health scoring and generation metadata.

## Module Structure

```
src/report/
├── __init__.py                    # Module initialization and exports (process_report)
├── processor.py                   # Step entry, file-level analysis, HTML/Markdown rendering
├── generator.py                   # Report file writers, summary/custom reports
├── analyzer.py                    # Pipeline data collection (collect_pipeline_data)
├── formatters.py                  # HTML/Markdown section rendering
├── pipeline_report.py             # Per-step status/timing/artifact sections
├── diff_report.py                 # Run-to-run diffing (compare_runs, archive_run)
├── model_family.py                # Model-family ledger markdown renderer
├── semantic_fidelity.py           # Semantic-fidelity ledger markdown renderer
├── cross_framework_reliability.py # Cross-framework reliability ledger renderer
├── mcp.py                         # MCP tool registrations
└── README.md                      # This documentation
```

## Core Components

### `process_report(target_dir: Path, output_dir: Path, verbose: bool = False, logger=None, **kwargs) -> bool`

Main entry point, called by `23_report.py` (Step 23). Determines the pipeline output directory (parent of `output_dir`) and delegates to `generator.generate_comprehensive_report()`.

- `report_formats` kwarg: subset of `["html", "markdown", "json"]` (default: all three)
- `include_performance`, `include_errors`, `include_dependencies` kwargs: section toggles (default True)

### `generator.generate_comprehensive_report(pipeline_output_dir, report_output_dir, logger, report_formats=None, include_performance=True, include_errors=True, include_dependencies=True) -> bool`

Collects data from all step directories via `analyzer.collect_pipeline_data()`, computes the pipeline health score, and writes the report files.

### `processor.generate_comprehensive_report(target_dir, output_dir, format="json", **kwargs) -> Dict[str, Any]`

File-level analysis of GNN `.md` files (`analyze_gnn_file()` per file); returns report data with `success`, `total_files`, `files_analyzed`, `summary`.

### Exports (`from report import ...`)

- `process_report`, `generate_report` (file-level report builder)
- `generate_comprehensive_report` (generator version)
- `analyze_gnn_file`, `generate_html_report`, `generate_markdown_report`
- `ReportGenerator`, `ReportFormatter`, `analyze_pipeline_data`
- `get_module_info`, `get_supported_formats`, `validate_report`

## Usage Examples

### Basic report generation

```python
from report import process_report
from pathlib import Path

success = process_report(
    target_dir=Path("output"),
    output_dir=Path("output/23_report_output"),
    verbose=True,
)
```

### Custom format selection

```python
success = process_report(
    target_dir=Path("output"),
    output_dir=Path("output/23_report_output"),
    report_formats=["html", "json"],
    include_dependencies=False,
)
```

## Integration with Pipeline

### Pipeline Step 23: Report Generation

`23_report.py` is a thin orchestrator delegating to `process_report()`. `main.py` also imports `report.pipeline_report.generate_pipeline_report` directly for step-level report sections.

### Output Structure

```
output/23_report_output/
├── comprehensive_analysis_report.html   # Full HTML analysis report
├── comprehensive_analysis_report.md     # Markdown analysis report
├── report_summary.json                  # Structured JSON export
├── report_generation_summary.json       # Generation metadata (health score, formats, options)
└── report_processing_summary.json       # Step processing summary
```

## Dependencies

- **Required (stdlib)**: pathlib, json, logging, typing
- **Optional**: none (HTML/Markdown generation is pure Python string templating)

## Testing

Tests live in `src/tests/report/` (integration, generation, formats, functional, diff/ledgers, MCP wrappers).

```bash
uv run --extra dev python -m pytest src/tests/report/ --cov=src/report
```

## References

- Project overview: ../../README.md
- Pipeline details: ../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
