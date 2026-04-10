# Specification: Generalized Notation Notation (GNN) Reference

## Design Requirements
The `doc/gnn` module does not contain execution logic mapping directly to the runtime graph. Instead, it serves as the **Static Source of Truth** for format specifications.

## Components
It governs the architectural parsing constraints across three main modalities:
1. **The Syntax Standard** (`gnn_syntax.md`): Defines variable encodings, connections mapping generative pipelines, and markup validation.
2. **The Output Target Frameworks**: How GNN compiles down into PyMDP, RxInfer, and JAX logic representation.
3. **Agent Integration** (`mcp` / `integration`): How external services can programmatically consume, augment, and output standard GNN files.

## Interfaces
GNN `.md` files are universally consumed starting from Step 3 (`3_gnn.py`) of the main pipeline. This entire documentation tree provides the exact syntax rules that Step 3 asserts against.
