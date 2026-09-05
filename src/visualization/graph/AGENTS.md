# Visualization Graph Sub-module

## Overview

Network graph visualization for GNN models. Generates directed and undirected graph layouts with ontology-aware labels and variable–parameter bipartite diagrams using networkx and matplotlib.

## Architecture

```
graph/
├── __init__.py                    # Package exports
├── network_visualizations.py      # Network graph generation
├── bipartite.py                   # Variable–parameter bipartite layouts
└── stats.py                       # Pure connection statistics (no plotting deps)
```

## Key Functions

- **`generate_network_visualizations(parsed_data, output_dir, model_name) -> List[str]`** — Creates network topology plots showing state-observation-action relationships with Active Inference edge semantics. Node colors come from `visualization.theme.VAR_TYPE_COLORS` and edge styles from `visualization.theme.get_edge_style` (single source of truth). Returns list of generated file paths.
- **`generate_variable_parameter_bipartite(parsed_data, output_dir, model_name) -> List[str]`** — Renders bipartite layouts: GNN variables (left) vs named parameter tensors (right), edges where parameter names match variable names.
- **`compute_connection_statistics(variables, connections) -> Dict[str, Any]`** (stats.py) — Pure degree-based statistics (totals, degree distribution, hubs, isolated nodes). Re-exported at the package root as `compute_connection_statistics` and pinned as `visualization._generate_network_statistics`.

### Internal Helpers

- `_var_type(var_info)` — Module-level alias of `visualization.compat.viz_compat.viz_var_type` (canonical).
- `_connection_is_undirected(conn_info)` — Detects undirected edge semantics.
- `_determine_connection_type(source_var, target_var, source_type=None, target_type=None)` — Maps Active Inference variable pairs to semantic connection types; every named type it emits has a style in `theme.EDGE_STYLES`.
- `_get_edge_style(connection_type)` — Thin delegate to `visualization.theme.get_edge_style` (the theme module owns the palette; do not fork styles here).
- `_compute_graph_metrics(variables, connections)` — Networkx-backed metrics: counts, type tallies, `gnn_edge_orientation`, density/clustering/connectivity.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: 3.2.0
**Last Updated**: 2026-09-04
