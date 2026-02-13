# GNN Formal Specifications

## Overview

This directory contains formal mathematical specifications of GNN (Generalized Notation Notation) models. Each file provides a rigorous, machine-checkable formalization of the GNN type system, Active Inference structures, and model validation properties in a different proof assistant or formal language.

## Purpose

The formal specifications serve three roles:

1. **Mathematical Foundation** — Provide precise definitions of GNN constructs (variables, connections, state spaces) that ground the Python implementation in `parser.py` and `types.py`.
2. **Property Verification** — Express and (where tooling permits) verify invariants such as matrix dimension consistency, probability normalization, and well-formedness of POMDP structures.
3. **Cross-Language Reference** — Offer templates for researchers who need to interface GNN models with theorem provers or symbolic computation environments.

## Specification Files

| File | Language | Focus |
|------|----------|-------|
| `isabelle.thy` | Isabelle/HOL | Set-theoretic types, well-formedness theorems |
| `lean.lean` | Lean 4 | Dependent types, category-theoretic model composition |
| `coq.v` | Coq | Dependent types, state inference and policy optimization proofs |
| `agda.agda` | Agda | Type-theoretic specification with constructive proofs |
| `alloy.als` | Alloy | Constraint-based model checking and structural verification |
| `z_notation.zed` | Z Notation | Schema-based specification with temporal logic |
| `tla_plus.tla` | TLA+ | Temporal logic specification of model dynamics |
| `maxima.mac` | Maxima | Symbolic computation and matrix operations |

## Relationship to Codebase

The Python dataclasses in `types.py` (`ParsedGNN`, `GNNVariable`, `GNNConnection`, `GNNFormat`) are the runtime counterparts of the structures formalized here. The `pomdp_extractor.py` module implements the POMDP extraction logic whose correctness properties are stated as theorems in the Isabelle, Lean, and Coq specifications.

## Usage

These files are primarily reference documents. To check a specification:

- **Isabelle**: Load `isabelle.thy` in Isabelle/jEdit
- **Lean**: `lake build` with `lean.lean` in a Lean 4 project
- **Coq**: `coqc coq.v`
- **Agda**: `agda agda.agda`
- **Alloy**: Open `alloy.als` in the Alloy Analyzer
- **Maxima**: `batch("maxima.mac")` in a Maxima session
