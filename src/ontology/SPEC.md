# Ontology Module Specification

## Overview
Ontology processing and validation for GNN models.

## Components

### Core
- `processor.py` - Ontology processor (pure composable primitives + pipeline entry points)
- `utils.py` - Module metadata helpers (`get_module_info`, `get_ontology_processing_options`, `get_mcp_interface`)
- `mcp.py` - MCP tool registration (4 tools, validated against the real vocabulary)
- `act_inf_ontology_terms.json` - Active Inference ontology vocabulary (64 canonical terms)

## Features
- Ontology term validation (case-insensitive)
- Semantic mapping
- Term extraction
- Nearest-term suggestion (Levenshtein + substring)
- In-memory vocabulary construction
- Prebuilt batch vocabulary index (`OntologyTermIndex`)
- Injectable vocabulary search paths (dependency injection)

## Key Exports
```python
from ontology import (
    process_ontology,
    analyze_ontology_content,
    validate_annotations,
    suggest_terms,
    summarise_coverage,
    build_ontology_terms,
    parse_annotation,
    ParsedAnnotation,
)
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
