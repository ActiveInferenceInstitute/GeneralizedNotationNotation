# CatColab Integration Documentation

Documentation for integrating GNN (Generalized Notation Notation) with CatColab, the Topos Institute's collaborative categorical modeling platform.

## Documents

| Document | Lines | Description |
|----------|-------|-------------|
| [AGENTS.md](AGENTS.md) | 55 | Agent scaffolding and navigation |
| [catcolab.md](catcolab.md) | 131 | Comprehensive CatColab platform overview (v0.4 Robin) |
| [catcolab_gnn.md](catcolab_gnn.md) | 352 | GNN-CatColab integration guide with structural mappings |

## Overview

CatColab provides category-theoretic compositional modeling with domain-specific logics. GNN provides Active Inference model specification and execution. Together they enable:

- **Formal modeling**: Categorical semantics for Active Inference agents
- **Compositional construction**: Build complex agents from verified subcomponents  
- **Cross-platform execution**: Model in CatColab, execute via GNN pipeline

## Key Integration Points

| GNN Component | CatColab Logic | Use Case |
|---------------|----------------|----------|
| A Matrix (Likelihood) | Schema | State-observation mapping |
| B Matrix (Transition) | Stock-and-Flow | Dynamics modeling |
| C Matrix (Preference) | Regulatory Network | Goal specification |
| State Space | Olog | Ontology definitions |
| Connections | Petri Net | Discrete transitions |

## Quick Links

- **CatColab Platform**: [catcolab.org](https://catcolab.org)
- **GNN Documentation**: [../gnn/README.md](../gnn/README.md)
- **DisCoPy Integration**: [../discopy/gnn_discopy.md](../discopy/gnn_discopy.md)
- **Pipeline Reference**: [../../src/AGENTS.md](../../src/AGENTS.md)

## Related Documentation

- [CatColab GitHub](https://github.com/ToposInstitute/CatColab)
- [AlgebraicJulia Ecosystem](https://www.algebraicjulia.org/)
- [Ontology System](../gnn/ontology_system.md)
