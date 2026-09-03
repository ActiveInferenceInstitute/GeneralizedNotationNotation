# Integration Module - Agent Scaffolding

## Module Overview

**Purpose**: System-level consistency validation — builds a NetworkX dependency graph of GNN model components, detects cycles and isolated components, verifies cross-file references, and runs meta-analysis over parameter-sweep execution outputs.

**Pipeline Step**: Step 17: Integration (17_integration.py)

**Category**: System Integration / Coordination

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-02

---

## Core Functionality

1. Build a directed dependency graph from GNN `## StateSpaceBlock` variables and `## Connections` edges
2. Detect short cycles (length <= 6, capped) and isolated components
3. Verify `$ref: name` cross-references resolve to a known component
4. Run the meta-analysis submodule over Step 12 execution outputs (when present)
5. Write `integration_results.json` and `integration_summary.md`

---

## API Reference

### Public Functions

#### `process_integration(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main integration processing function called by orchestrator (17_integration.py). Builds the system dependency graph, runs consistency checks, and performs meta-analysis of parameter sweeps.

**Parameters**:
- `target_dir` (Path): Directory containing pipeline outputs to integrate (GNN `.md` files)
- `output_dir` (Path): Output directory for integration results
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Accepted and ignored; reserved for pipeline-template compatibility

**Returns**: `bool` - True if integration processing succeeded, False otherwise

**Example**:
```python
from integration import process_integration
from pathlib import Path

success = process_integration(
    target_dir=Path("output"),
    output_dir=Path("output/17_integration_output"),
    verbose=True,
)
```


#### Module Coordination (via `process_integration`)

Graph construction, cycle detection, and cross-reference validation are performed internally by `process_integration()`. The `integration_results.json` report includes:
- `processed_files` (int): Number of GNN files scanned
- `system_graph_stats` (Dict): nodes, edges, cycles, isolated_nodes, components
- `issues` (List[str]): Isolated components, undefined `$ref:` references, suspicious type names
- `meta_analysis` (Dict): Sweep analysis results (only when execution outputs exist)

---

## Dependencies

### Required Dependencies
- `pathlib`, `typing`, `logging` - Standard library

### Optional Dependencies
- `networkx` - Graph construction and cycle/isolate analysis (dict-based fallback without it)
- `matplotlib` - Meta-analysis visualizations

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing patterns
- `pipeline.config` - Pipeline configuration management

---

## Configuration

### Environment Variables

None dedicated to this module. Integration behavior is fixed by
`process_integration()` in `integration/processor.py`; output location is set
via the `--output-dir` CLI flag on `17_integration.py` and `input/config.yaml`
pipeline settings.

---

## Usage Examples

### Basic Usage
```python
from integration.processor import process_integration

success = process_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/17_integration_output"),
    verbose=True,
)
```

---

## Output Specification

### Output Products
- `integration_results/integration_results.json` - Graph statistics, issues, and meta-analysis results
- `integration_results/integration_summary.md` - Human-readable integration summary
- `integration_results/meta_analysis/` - Sweep plots, validation, statistics, report (when execution outputs exist)

### Output Directory Structure
```
output/17_integration_output/
└── integration_results/
    ├── integration_results.json
    ├── integration_summary.md
    └── meta_analysis/
```

---

## Performance Characteristics

### Performance
- **Fast Path**: <1s for basic graph validation
- **Analysis Depth**: O(N+E) complexity for cycle detection
- **Memory**: Proportional to graph size (Node/Edge count)

---

## Error Handling

### Graceful Degradation
- Per-file parse failures are logged and skipped
- Graph-analysis failures degrade to dict-based edge counting (no NetworkX)
- Meta-analysis failures are warnings only and never fail the step

### Error Categories
1. **Parse Errors**: Malformed GNN file (skipped with warning)
2. **Graph Errors**: Graph analysis failure (fallback statistics)
3. **Reference Errors**: Undefined `$ref:` or suspicious type names (reported as issues)

---

## Integration Points

### Orchestrated By
- **Script**: `17_integration.py` (Step 17) - thin orchestrator delegating to `process_integration()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `src/tests/integration/test_integration_overall.py` - Module-level integration tests
- `src/main.py` - Runs `17_integration.py` as a pipeline step (subprocess)

### Data Flow
```
GNN Files (target_dir) → Graph Construction → Consistency Checks → integration_results.json + integration_summary.md
Step 12 Execute Outputs → Meta-Analysis → meta_analysis/ artifacts
```

---

## Testing

### Test Files
- `src/tests/integration/test_integration_functional.py` - Functional integration tests
- `src/tests/integration/test_integration_processor.py` - Processor-level integration tests
- `src/tests/integration/test_integration_mcp.py`, `test_integration_mcp_tools.py` - MCP tool tests
- `src/tests/integration/test_meta_analysis_none_states.py` - Meta-analysis edge cases

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/integration/ \
    --cov=src/integration --cov-report=term-missing
```

### Key Test Scenarios
1. Graph construction and consistency checks across step combinations
2. Behavior when NetworkX or execution outputs are unavailable
3. Meta-analysis record extraction accuracy
4. Error handling with malformed GNN files

---

## MCP Integration

### Tools Registered
- `process_integration` - Run GNN integration processing
- `list_supported_integrations` - List integration targets and availability
- `get_integration_status` - Inspect artifacts from a previous integration run
- `check_integration_dependencies` - Report optional integration dependencies

Each name above is registered by `register_tools()` with a named callable,
JSON input schema, module/category metadata, and explicit success/error results.

### MCP File Location
- `src/integration/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Cross-file dependencies not detected
**Symptom**: Expected cross-file dependencies are absent from the graph
**Cause**: Cross-file reference edges via content matching were deliberately removed (models share vocabulary like `s_prime`/`beta`, causing false positives)
**Solution**: Use explicit `$ref: name` syntax for real cross-file dependencies

#### Issue 2: Cross-reference validation errors
**Symptom**: Valid references reported as missing
**Cause**: `$ref:` name not declared in any scanned GNN file, or file path resolution issues
**Solution**:
- Ensure the referenced component is declared in a `## StateSpaceBlock` section
- Check reference format matches the `$ref: name` pattern
- Ensure referenced files are in `target_dir` or a discoverable `input/gnn_files/`

---

## Version History

### Current Version: 1.6.0 (module `__init__.py`), pipeline release 3.2.0

**Features**:
- Dependency graph construction (NetworkX)
- Cycle and isolated-component detection
- `$ref:` cross-reference validation
- Meta-analysis of parameter sweeps (`integration/meta_analysis/`)

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced dependency analysis
- **Future**: Real-time integration monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Pipeline Configuration](../pipeline/AGENTS.md)

### External Resources
- [NetworkX Documentation](https://networkx.org/)

---

**Last Updated**: 2026-09-02
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
