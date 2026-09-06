# Integration Module Specification

System-level consistency validation for GNN pipeline steps: dependency-graph construction (NetworkX), cycle/isolated-component detection, `$ref:` cross-reference validation, and meta-analysis of parameter-sweep execution outputs.

## Components

### Core
- `parsing.py` - Pure GNN text-extraction primitives (sections, variables, connections, refs, types)
- `graph.py` - `build_system_graph()`, `verify_references()`, `analyze_system()`, `export_dependency_graph()`; `SystemAnalysis`/`SystemGraphStats` dataclasses
- `processor.py` - `process_integration()`: composes the above into the Step-17 report artifacts
- `meta_analysis/` - Parameter sweep runtime and simulation analysis (collector, statistics, validator, visualizer, reporter)
- `mcp.py` - MCP tool registrations (4 tools)

## Key Exports
```python
from gnn.integration import (
    process_integration,
    analyze_system,
    build_system_graph,
    verify_references,
    export_dependency_graph,
    SystemAnalysis,
    SystemGraphStats,
    run_meta_analysis,
    SweepDataCollector,
    SweepRecord,
)
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
