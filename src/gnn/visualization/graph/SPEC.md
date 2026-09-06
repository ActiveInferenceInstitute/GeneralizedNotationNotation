# Graph Visualization — Technical Specification

**Version**: 1.6.0

## Graph Types

- **Network topology** — Full model connectivity (states, observations, actions)
- **Bipartite layout** — Separated hidden states and observations

## Layout Algorithm

- Spring layout (default) for network graphs
- Bipartite layout for two-set separation

## Edge Semantics

- Solid arrows: deterministic transitions
- Dashed arrows: probabilistic transitions
- Edge width: proportional to transition probability

## Dependencies

- `networkx >= 3.0` (required)
- `matplotlib >= 3.5` (required for rendering)
