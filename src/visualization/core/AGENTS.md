# Visualization Core Sub-module

## Overview

Core orchestration logic for Step 8 (Visualization). Handles JSON-first model loading from Step 3 parsed output and coordinates graph and matrix visualization generation.

## Architecture

```
core/
├── __init__.py          # Package exports (9 lines)
├── process.py           # Step-8 visualization orchestration (286 lines)
└── parsed_model.py      # Parsed model data structures (133 lines)
```

## Key Functions

- **`process_visualizations(target_dir, output_dir)`** — Main orchestration entry point; loads parsed JSON models and dispatches to graph and matrix visualizers.
- **`ParsedModel`** — Data class representing a loaded GNN model with states, observations, connections, and matrices.
- **JSON-first loading** — Prefers `*_parsed.json` files from Step 3 output for reliable structured data access.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
