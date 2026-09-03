# Report Module Specification

Consolidation of pipeline outputs into comprehensive HTML/Markdown/JSON analysis reports with health scoring. Step 23 of the GNN pipeline.

## Components

### Core
- `processor.py` - `process_report()` step entry, `analyze_gnn_file()`, HTML/Markdown renderers
- `generator.py` - `generate_comprehensive_report()` and report file writers (HTML/Markdown/JSON/summary/custom)
- `analyzer.py` - `collect_pipeline_data()` aggregation over step output directories
- `formatters.py` - HTML/Markdown section rendering (performance, errors, steps, visualizations)
- `pipeline_report.py` - Per-step status/timing/artifact/statistics sections
- `diff_report.py` - `compare_runs()` / `archive_run()` run-to-run diffing
- `model_family.py`, `semantic_fidelity.py`, `cross_framework_reliability.py` - ledger markdown renderers
- `mcp.py` - MCP tool registrations (5 tools)

## Features
- Multi-format report generation (HTML, Markdown, JSON)
- Pipeline health score (0-100)
- Run-to-run diff reports and acceptance-ledger renderers

## Key Exports
```python
from report import process_report
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
