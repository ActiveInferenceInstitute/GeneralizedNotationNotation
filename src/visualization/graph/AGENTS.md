# visualization.graph

| Symbol | Location |
|--------|----------|
| `generate_network_visualizations` | `network_visualizations.py` — directed vs undirected edges, ontology node labels |
| `generate_variable_parameter_bipartite` | `bipartite.py` — variables vs parameter names |

Uses `advanced_visualization._shared.normalize_connection_format`. `{model}_network_stats.json` includes `gnn_edge_orientation` (directed vs undirected pair counts). Root `network_visualizations.py` re-exports `generate_network_visualizations`.
