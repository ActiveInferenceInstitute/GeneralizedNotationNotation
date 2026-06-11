# Specification: Advanced GNN Documentation

## Scope
The `doc/gnn/advanced/` subtree documents GNN features and patterns that
sit above the core specification: multi-agent models, hierarchical inference,
neurosymbolic LLM integration, and the Active Inference ontology system.

## Contents
| File | Purpose |
|------|---------|
| `README.md` | Entry point + navigation |
| `advanced_modeling_patterns.md` | Patterns: hierarchical, multi-modal, coupled agents |
| `gnn_multiagent.md` | Multi-agent GNN specification and encoding |
| `gnn_ontology.md` | Active Inference ontology mapping rules |
| `ontology_system.md` | Ontology system architecture |
| `gnn_llm_neurosymbolic_active_inference.md` | LLM + Active Inference integration |

## Versioning
All documents in this subtree inherit the bundle version from
[`doc/gnn/SPEC.md`](../SPEC.md). Individual docs may pin specific feature
versions in their frontmatter when describing pre-release extensions.

## Status
Maintained. Cross-referenced from `doc/gnn/README.md` and
`doc/gnn/AGENTS.md`. Every document links back to the parent index.
