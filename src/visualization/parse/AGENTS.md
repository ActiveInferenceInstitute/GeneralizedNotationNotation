# Visualization Parse Sub-module

## Overview

GNN file parsing utilities specific to the visualization pipeline. Provides markdown and raw GNN file parsing to extract model structure for visualization when pre-parsed JSON is unavailable.

## Architecture

```
parse/
├── __init__.py            # Package exports (4 lines)
├── gnn_file_parser.py     # GNN file structure extraction (250 lines)
└── markdown.py            # Markdown GNN format parsing (203 lines)
```

## Key Functions

- **`parse_gnn_for_viz(file_path)`** — Extracts visualization-relevant model structure (states, observations, connections, matrices) from raw GNN files.
- **`parse_markdown_gnn(file_path)`** — Specialized parser for the GNN markdown format, extracting section-delimited model components.
- **Fallback role** — Used when Step 3 JSON output is unavailable, providing degraded but functional visualization.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
