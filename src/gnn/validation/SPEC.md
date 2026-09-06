# Validation Module Specification

## Overview
GNN model validation including semantic, structural, and mathematical validation.

## Components

### Core
- `__init__.py` - `process_validation()` binding, module exports, metadata
- `workflow.py` - Directory-level validation workflow (`validate_directory`, injected `StageServices`)
- `structure.py` - Shared helpers: content extraction, Tarjan cycle detection, score clamping
- `semantic_validator.py` - Semantic validation with mapping support (`validate_content` for no-I/O use)
- `performance_profiler.py` - Complexity, memory, and parallelization estimation
- `consistency_checker.py` - Naming, style, structure, and reference consistency
- `mcp.py` - MCP tool registration

## Validation Levels
- `basic` - Structure checks
- `standard` - Connection integrity
- `strict` - Active Inference principles
- `research` - Advanced mathematical properties

## Receipt and Outcome Contract

- Current-pass success requires a nonempty manifest with every file successful; semantic invalidity and recovery fail the pass.
- `run_id` overrides manifest `run_id` / `timestamp`; manifests without either are unbound records scoped to the output directory.
- Distinct current source paths replace prior entries within that run/configuration. Input hashes and configuration identify each receipt; aggregate counts and averages are recalculated.
- `validation_results.json` includes aggregate `summary` plus `current_summary`; `validation_summary.json` remains aggregate.
- `validate_content` provides common semantic evidence for CLI, validation MCP, and the pipeline while their exit/transport policies remain separate.

## Mapping Types Supported
`identity`, `transpose`, `reshape`, `broadcast`, `reduce`

## Key Exports
```python
from gnn.validation import (
    process_validation,
    SemanticValidator,
    validate_content,
    validate_directory,
    StageServices,
)
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
