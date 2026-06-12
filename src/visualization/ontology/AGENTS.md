# Visualization Ontology Sub-module

## Overview

Ontology-specific visualization for Active Inference terminology. Generates visual mappings between GNN model elements and Active Inference ontology concepts.

## Architecture

```
ontology/
├── __init__.py       # Package exports (3 lines)
└── visualizer.py     # Ontology visualization engine (240 lines)
```

## Key Functions

- **`visualize_ontology_mappings(model, ontology, output_dir)`** — Creates visual diagrams showing how model variables map to Active Inference concepts.
- **Term highlighting** — Color-codes model elements by ontology category (beliefs, observations, actions, preferences).
- **Coverage reporting** — Shows which ontology terms are present and which are missing from the model.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
