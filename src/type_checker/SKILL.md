---
name: gnn-type-checking
description: GNN syntax validation and resource estimation. Use when checking GNN model types, validating matrix dimensions, verifying state space consistency, or estimating computational resources for model execution.
---

# GNN Type Checking (Step 5)

## Purpose

Validates GNN model syntax, checks matrix dimension consistency, verifies type annotations, and estimates computational resource requirements for model execution.

## Key Commands

```bash
# Run type checking
python src/5_type_checker.py --target-dir input/gnn_files --output-dir output --verbose

# Strict mode
python src/main.py --only-steps 5 --strict

# Fast validation only
python src/main.py --only-steps 5 --verbose
```

## API

```python
from type_checker import GNNTypeChecker, estimate_file_resources

# Create type checker and validate a file
checker = GNNTypeChecker()
result = checker.check("path/to/model.md")

# Estimate resources for a file
resources = estimate_file_resources("path/to/model.md")
```

## Key Exports

- `GNNTypeChecker` — class providing syntax validation and type checking
- `estimate_file_resources` — estimates computational requirements for a GNN file

## Validation Checks

| Check | Description |
| ------- | ------------- |
| **Dimension consistency** | Matrix dimensions match across connections |
| **Type compatibility** | Connected nodes have compatible types |
| **Required sections** | All mandatory GNN sections present |
| **Parameterization** | Initial values match declared dimensions |
| **Ontology annotations** | Annotations reference valid ontology terms |

## Output

- Type checking report in `output/5_type_checker_output/`
- Dimension validation results
- Resource estimation summaries


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `validate_gnn_files`
- `validate_single_gnn_file`

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
