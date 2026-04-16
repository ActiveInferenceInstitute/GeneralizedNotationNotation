# Visualization Graph Sub-module

## Overview

Network graph visualization for GNN models. Generates directed and undirected graph layouts with ontology-aware labels using networkx and matplotlib.

## Architecture

```
graph/
├── __init__.py                    # Package exports (4 lines)
├── network_visualizations.py      # Network graph generation (507 lines)
└── bipartite.py                   # Bipartite graph layouts (125 lines)
```

## Key Functions

- **`generate_network_graph(model, output_dir)`** — Creates network topology plots showing state-observation-action relationships.
- **`generate_bipartite_graph(model, output_dir)`** — Renders bipartite layouts separating hidden states from observations.
- **Ontology labels** — Applies Active Inference ontology terms to graph node labels when available.
- **Edge typing** — Distinguishes directed (causal) from undirected (correlational) edges.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
