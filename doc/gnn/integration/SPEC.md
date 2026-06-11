# Specification: Integration Documentation

## Scope
Guides for integrating GNN with external tooling: cross-framework code
generation, visualization tooling, export pipelines, and orchestration
with other Active Inference stacks.

## Contents
| File | Purpose |
|------|---------|
| `framework_integration_guide.md` | How to target GNN from a new framework backend |
| `gnn_export.md` | Export formats (JSON, XML, GraphML, GEXF, pickle) and conversion rules |
| `gnn_implementation.md` | Implementation notes for Python / Julia integrators |
| `gnn_visualization.md` | Visualization pipeline integration |

## Status
Maintained. When adding a new rendering/execution backend, cross-reference
`framework_integration_guide.md` to confirm the new backend follows the
published contract.
