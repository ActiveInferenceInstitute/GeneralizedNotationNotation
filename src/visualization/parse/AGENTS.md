# Visualization Parse Sub-module

## Overview

GNN file parsing utilities specific to the visualization pipeline. Provides markdown GNN format parsing to extract model structure for visualization when pre-parsed JSON from Step 3 is unavailable.

## Architecture

```
parse/
├── __init__.py            # Package exports
├── gnn_file_parser.py     # GNN file structure extraction
└── markdown.py            # Markdown GNN format parsing
```

## Key Functions

### markdown.py

- **`parse_gnn_content(content: str) -> Dict[str, Any]`** — Parse raw GNN markdown content into structured dict with `variables`, `connections`, `parameters`, `raw_sections`, and `ActInfOntologyAnnotation`.

### Internal Helpers

- `_is_complete_parameter(value_str)` — Detect multi-line parameter value completeness.
- `_parse_parameter_value(value_str)` — Safely parse parameter value strings (JSON, lists, scalars).
- `_save_parameter(parsed, param_name, param_lines)` — Accumulate multi-line parameter definitions.

## Raw Markdown Parser Role

This module provides the explicit raw-Markdown parser for the visualization
pipeline. The preferred path loads structured JSON from Step 3 via
`core/parsed_model.py`. When Step 3 output is unavailable,
`parse_gnn_content` provides reduced but functional model data for
visualization.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
**Last Updated**: 2026-05-12
