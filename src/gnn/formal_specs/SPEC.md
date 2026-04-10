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
3. **Interchange**: Languages that have Python parsers/serializers (Lean, Coq, Agda, TLA+, Alloy, Z, Maxima, BNF) participate in the **`gnn/parsers/`** pipeline. **EBNF** shares **`GrammarSerializer`** with **BNF** and is not always a separate row in `test_round_trip.py`.

## Testing

Automated round-trip coverage for formal *file* formats follows **`test_round_trip.py`** `FORMAT_TEST_CONFIG` (e.g. Agda, TLA+, Alloy, Z, BNF — not necessarily every formal artifact in this directory on every run). See **[../SPEC.md](../SPEC.md)** and **`../testing/README_round_trip.md`**.
