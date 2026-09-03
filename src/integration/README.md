# Integration Module

This module performs system-level consistency checks for the GNN pipeline: it builds a dependency graph of model components (NetworkX), detects circular dependencies and isolated components, verifies cross-file references, and runs a meta-analysis over parameter-sweep execution outputs.

## Module Structure

```
src/integration/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # process_integration(): graph build + checks
├── meta_analysis/                 # Parameter sweep runtime/simulation analysis
├── mcp.py                         # MCP tool registrations
└── README.md                      # This documentation
```

## Core Components

### `process_integration(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`

Main entry point, called by `17_integration.py` (Step 17).

- Scans `target_dir` (and, walking upward, any `input/gnn_files/`) for GNN `.md` files
- Extracts variables from `## StateSpaceBlock` sections and edges from `## Connections` (`>`, `-`, `<` operators) into a NetworkX directed graph (falls back to plain dicts if NetworkX is absent)
- Counts short cycles (length <= 6, capped at 500 or a 5 s budget) as informational structure — intra-model cycles are expected mathematical relationships, not errors
- Flags isolated components (no connections) as issues
- Verifies `$ref: name` references resolve to a known component; flags capitalized `type:` values that are not built-ins
- Runs the meta-analysis (below) when `12_execute_output/` exists, non-fatal on failure
- Writes `integration_results.json` and `integration_summary.md`

**Returns:** `bool` — True if processing succeeded.

### Meta-Analysis (`integration/meta_analysis/`)

Analyzes parameter-sweep runtime and simulation outputs produced by Step 12 (and Step 11 render metadata when available):

- Runtime scaling analysis across (N, T) parameter grids
- Cross-framework performance comparison heatmaps
- Simulation metric extraction (VFE, EFE, belief accuracy)
- Scaling-law regression and markdown report generation

Submodules: `collector.py`, `statistics.py`, `validator.py`, `visualizer.py`, `reporter.py`.

### Exports (`from integration import ...`)

- `process_integration`
- `run_meta_analysis`, `SweepDataCollector`, `SweepRecord`
- `FEATURES`, `__version__`

## Usage Examples

### Basic usage

```python
from integration import process_integration
from pathlib import Path

success = process_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/17_integration_output"),
    verbose=True,
)
```

### Meta-analysis API

```python
from integration.meta_analysis import run_meta_analysis

results = run_meta_analysis(
    execute_output_dir=Path("output/12_execute_output"),
    output_dir=Path("output/17_integration_output/integration_results/meta_analysis"),
    render_output_dir=Path("output/11_render_output"),
    logger=logger,
)
```

## Integration with Pipeline

### Pipeline Step 17: System Integration

`17_integration.py` is a thin orchestrator: it parses `--target-dir`, `--output-dir`, `--recursive`, and `--verbose` arguments (standardized pipeline template) and delegates to `process_integration()`.

### Output Structure

```
output/17_integration_output/
└── integration_results/
    ├── integration_results.json    # Graph stats, issues, meta-analysis results
    ├── integration_summary.md      # Human-readable summary
    └── meta_analysis/              # Sweep plots, validation, statistics, report
                                   # (only when execution outputs exist)
```

### Error Handling

`process_integration()` is fail-soft: per-file parse failures are logged and skipped, graph-analysis and meta-analysis failures are warnings, and unexpected exceptions are logged with a False return.

## Dependencies

### Required
- `pathlib`, `logging`, `json`, `re` (stdlib)

### Optional
- `networkx` — graph construction and cycle/isolate analysis; without it the module falls back to plain dict-based edge counting with reduced statistics
- `matplotlib` — meta-analysis visualizations

## Testing and Validation

Tests live in `src/tests/integration/` (`test_integration_functional.py`, `test_integration_processor.py`, `test_integration_overall.py`, MCP tests, and meta-analysis tests).

```bash
uv run --extra dev python -m pytest src/tests/integration/ --cov=src/integration
```

## Troubleshooting

### False-positive cross-file dependencies
Cross-file edges via substring matching were removed deliberately: GNN models share a common mathematical vocabulary (`s_prime`, `beta`, `alpha`), so content matching always generated false-positive edges. Use explicit `$ref: name` syntax for real cross-file dependencies.

### Meta-analysis skipped
The meta-analysis only runs when `output/12_execute_output/` (Step 12 execution outputs) exists. No execution outputs means the sweep analysis is skipped with an informational log — this is expected, not an error.

## References

- Project overview: ../../README.md
- Pipeline details: ../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
