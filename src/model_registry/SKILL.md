---
name: gnn-model-registry
description: GNN model versioning and registry management. Use when registering parsed models, tracking model versions, querying model metadata (author, license, version), or managing the model catalog.
---

# GNN Model Registry (Step 4)

## Purpose

Manages a registry of parsed GNN models with versioning, metadata extraction, and catalog management. Tracks model lineage including author, license, and version information.

## Key Commands

```bash
# Run model registry step
python src/4_model_registry.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 4 --verbose
```

## API

```python
from model_registry import ModelRegistry, process_model_registry

# Use the registry class
registry = ModelRegistry()

# Process model registry step (used by pipeline)
result = process_model_registry(target_dir, output_dir, verbose=True)
```

## Key Exports

- `ModelRegistry` — class managing model catalog, versioning, and metadata
- `process_model_registry` — main processing function for pipeline integration

## Metadata Extraction

The registry automatically extracts:

- **Author** and **license** from file headers
- **Version** information from model definitions
- **Dimensions** and **types** from StateSpaceBlock
- **Framework compatibility** from annotations

## Output

- Model catalog in `output/4_model_registry_output/`
- JSON registry index
- Version history tracking


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `model_registry.register_model`
- `model_registry.get_model`
- `model_registry.search_models`

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
