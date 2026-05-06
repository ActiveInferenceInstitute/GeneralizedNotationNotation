---
name: gnn-validation
description: GNN advanced validation and consistency checking. Use when performing deep validation of GNN models, checking cross-model consistency, verifying structural integrity, or running validation reports.
---

# GNN Validation (Step 6)

## Purpose

Performs advanced validation beyond basic type checking, including semantic validation, performance profiling, consistency checking, and comprehensive validation reporting.

## Key Commands

```bash
# Run validation
python src/6_validation.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 6 --verbose
```

## API

```python
from validation import (
    SemanticValidator, process_semantic_validation,
    PerformanceProfiler, profile_performance,
    ConsistencyChecker, check_consistency,
    process_validation
)

# Run full validation (used by pipeline)
result = process_validation(target_dir, output_dir, verbose=True)

# Semantic validation
validator = SemanticValidator()
result = process_semantic_validation(parsed_model)

# Consistency checking
checker = ConsistencyChecker()
result = check_consistency(parsed_models)

# Performance profiling
profiler = PerformanceProfiler()
result = profile_performance(parsed_model)
```

## Key Exports

- `SemanticValidator` / `process_semantic_validation` — Active Inference constraint validation
- `PerformanceProfiler` / `profile_performance` — model performance analysis
- `ConsistencyChecker` / `check_consistency` — cross-model coherence checks
- `process_validation` — main pipeline processing function

## Output

- Validation reports in `output/6_validation_output/`
- Per-model validation summaries
- Cross-model consistency analysis


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `check_schema_compliance`
- `get_validation_report`
- `process_validation`
- `validate_gnn_file`

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
