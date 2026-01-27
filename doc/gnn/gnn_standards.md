# GNN Domain Knowledge and Standards

## Pipeline Processing Standards

The GNN pipeline follows strict architectural patterns and standards:

- **Thin Orchestrator Pattern**: All 25 pipeline steps delegate to modular implementations
  - See: **[src/README.md](../../src/README.md)** for thin orchestrator pattern details
- **Module Architecture**: Each module follows consistent structure with public APIs
  - See: **[src/AGENTS.md](../../src/AGENTS.md)** for complete module registry
- **Testing Standards**: No mocks, real data validation, >90% test coverage
  - See: **[doc/gnn/REPO_COHERENCE_CHECK.md](REPO_COHERENCE_CHECK.md)** for quality standards

**Architecture Documentation:**

- [architecture_reference.md](architecture_reference.md): Implementation patterns and data flow
- [src/README.md](../../src/README.md): Pipeline safety and reliability

---

### GNN File Structure Understanding

- **GNN Files**: Markdown-based (.md) with specific sections:
  - `GNNVersionAndFlags`: Version specification and processing flags
  - `ModelName`: Descriptive model identifier
  - `ModelAnnotation`: Free-text explanation of model purpose and features
  - `StateSpaceBlock`: Variable definitions with dimensions/types (s_fX[dims,type])
  - `Connections`: Directed/undirected edges showing dependencies (>, -, ->)
  - `InitialParameterization`: Starting values, matrices (A, B, C, D), priors
  - `Equations`: LaTeX-rendered mathematical relationships
  - `Time`: Temporal settings (Dynamic/Static, DiscreteTime, ModelTimeHorizon)
  - `ActInfOntologyAnnotation`: Mapping to Active Inference Ontology terms
  - `Footer` and `Signature`: Provenance information

### GNN Syntax and Punctuation

- **Variables**: Use underscore for subscripts (X_2), caret for superscripts (X^Y)
- **Dimensions**: Square brackets for array dimensions [2,3] = 2x3 matrix
- **Causality**: `>` for directed edges (X>Y), `-` for undirected (X-Y)
- **Operations**: Standard math operators (+, -, *, /, |)
- **Grouping**: Parentheses (), exact values {1}, indexing/dimensions [2,3]
- **Comments**: Triple hashtags (###) for inline comments
- **Probability**: Conditional probability notation P(X|Y) using pipe |
