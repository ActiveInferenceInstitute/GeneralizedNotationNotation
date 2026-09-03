# Research Module Specification

Deterministic rule-based static analysis of GNN models with experimental hypothesis generation. Step 19 of the GNN pipeline.

## Components

### Core
- `processor.py` - `process_research()` step entry; `detect_model_family()`, `extract_state_space_dims()`, `count_connections()`, `generate_rule_based_hypotheses()` rule engine
- `mcp.py` - MCP tool registrations (4 tools)

## Features
- Rule-based hypothesis generation with evidence justification
- Model-family detection and structural diagnostics
- Optional LLM-powered hypothesis enrichment (degrades to rule-based without a provider)

## Key Exports
```python
from research import process_research
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
