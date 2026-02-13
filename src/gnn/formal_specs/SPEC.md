# GNN Formal Specifications — Specification

## Architecture

This module provides **static formal specification files** in 8 proof/specification languages. Each file formalizes the GNN model structure and can be independently verified using its respective proof assistant.

## Components

| Language | Proof System | Verification Type |
|----------|-------------|-------------------|
| Isabelle/HOL | Isabelle | Higher-order logic proofs |
| Lean 4 | Lean | Dependent type checking |
| Coq | Coq | Constructive proofs |
| Agda | Agda | Dependent types |
| Alloy | Alloy Analyzer | Bounded model checking |
| Z-notation | Z tools | Set-theoretic analysis |
| TLA+ | TLC | Temporal model checking |
| Maxima | wxMaxima | Symbolic computation |

## Requirements

1. **Completeness**: Each spec must formalize the core GNN types (`StateSpaceBlock`, `Connections`, `Parameters`, `Equations`)
2. **Consistency**: All 8 specs must describe the same structural invariants
3. **Round-trip**: Each spec format must support serialization and parsing via `gnn/parsers/`

## Testing

Round-trip fidelity is verified by `gnn/testing/test_round_trip.py`, which tests all formal spec formats.
