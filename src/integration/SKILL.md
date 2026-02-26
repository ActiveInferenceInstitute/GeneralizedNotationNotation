---
name: gnn-system-integration
description: GNN system integration and cross-module coordination. Use when coordinating data flow between pipeline steps, resolving cross-module dependencies, or configuring inter-step communication.
---

# GNN System Integration (Step 17)

## Purpose

Coordinates data flow and communication between pipeline modules. Manages cross-module dependencies, resolves data handoffs, and ensures consistent state across the 25-step pipeline.

## Key Commands

```bash
# Run integration step
python src/17_integration.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 17 --verbose
```

## API

```python
from integration import process_integration

# Process integration step (used by pipeline)
result = process_integration(target_dir, output_dir, verbose=True)
```

## Key Exports

- `process_integration` — main pipeline processing function

## Integration Points

| Source Step | Target Step | Data Flow |
| ------------ | ------------- | ----------- |
| Step 3 (GNN) | Steps 5–8, 10–11, 13 | Parsed models |
| Step 11 (Render) | Step 12 (Execute) | Generated scripts |
| Step 12 (Execute) | Step 16 (Analysis) | Execution results |
| Step 16 (Analysis) | Step 23 (Report) | Analysis summaries |

## Output

- Integration reports in `output/17_integration_output/`
- Cross-module dependency resolution logs


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_integration`
- `list_supported_integrations`
- `get_integration_status`

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
