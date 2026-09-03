# Integration Module Specification

System-level consistency validation for GNN pipeline steps: dependency-graph construction (NetworkX), cycle/isolated-component detection, `$ref:` cross-reference validation, and meta-analysis of parameter-sweep execution outputs.

## Components

### Core
- `processor.py` - `process_integration()`: graph build, consistency checks, report writing
- `meta_analysis/` - Parameter sweep runtime and simulation analysis (collector, statistics, validator, visualizer, reporter)
- `mcp.py` - MCP tool registrations (4 tools)

## Key Exports
```python
from integration import process_integration, run_meta_analysis, SweepDataCollector, SweepRecord
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
