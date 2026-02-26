---
name: gnn-research-tools
description: GNN research tools and experimental features. Use when running experimental analyses, prototyping new pipeline features, conducting research experiments on GNN models, or exploring novel Active Inference patterns.
---

# GNN Research Tools (Step 19)

## Purpose

Provides experimental and research-oriented tools for exploring novel Active Inference patterns, prototyping new pipeline capabilities, and conducting research experiments on GNN model collections.

## Key Commands

```bash
# Run research tools
python src/19_research.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 19 --verbose
```

## API

```python
from research import process_research

# Process research step (used by pipeline)
result = process_research(target_dir, output_dir, verbose=True)
```

## Key Exports

- `process_research` — main pipeline processing function

## Capabilities

- **Experimental analysis**: Novel analysis methods in development
- **Prototyping**: Rapid prototyping of new pipeline features
- **Pattern exploration**: Discover patterns across model collections
- **Benchmarking**: Performance benchmarks for pipeline operations

## Output

- Research outputs in `output/19_research_output/`
- Experimental analysis results


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_research`
- `list_research_topics`
- `read_research_results`

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
