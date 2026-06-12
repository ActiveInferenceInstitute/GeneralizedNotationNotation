# Visualization Graph Sub-module

## Overview

Network graph visualization for GNN models. Generates directed and undirected graph layouts with ontology-aware labels and variable–parameter bipartite diagrams using networkx and matplotlib.

## Architecture

```
graph/
├── __init__.py                    # Package exports
├── network_visualizations.py      # Network graph generation
└── bipartite.py                   # Variable–parameter bipartite layouts
```

## Key Functions

- **`generate_network_visualizations(parsed_data, output_dir, model_name) -> List[str]`** — Creates network topology plots showing state-observation-action relationships with Active Inference edge semantics. Returns list of generated file paths.
- **`generate_variable_parameter_bipartite(parsed_data, output_dir, model_name) -> List[str]`** — Renders bipartite layouts: GNN variables (left) vs named parameter tensors (right), edges where parameter names match variable names.

### Internal Helpers

- `_var_type(var_info)` — Delegates to `visualization.compat.viz_compat.viz_var_type` (canonical).
- `_connection_is_undirected(conn_info)` — Detects undirected edge semantics.
- `_determine_connection_type(source_var, target_var, conn_info)` — Maps Active Inference variable pairs to semantic connection types.
- `_get_edge_style(connection_type)` — Returns color/width/alpha/style dict for a connection type (professional hex palette).
- `_generate_network_statistics(variables, connections)` — Computes node/edge counts, degree distribution, `gnn_edge_orientation`.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 1.6.0
**Last Updated**: 2026-05-12
