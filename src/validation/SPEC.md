# Validation Module Specification

## Overview
GNN model validation including semantic, structural, and mathematical validation.

## Components

### Core
- `__init__.py` - `process_validation()` orchestrator and module exports
- `semantic_validator.py` - Semantic validation with mapping support
- `performance_profiler.py` - Complexity, memory, and parallelization estimation
- `consistency_checker.py` - Naming, style, structure, and reference consistency
- `mcp.py` - MCP tool registration

## Validation Levels
- `basic` - Structure checks
- `standard` - Connection integrity
- `strict` - Active Inference principles
- `research` - Advanced mathematical properties

## Mapping Types Supported
`identity`, `transpose`, `reshape`, `broadcast`, `reduce`

## Key Exports
```python
from validation import process_validation, SemanticValidator
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
