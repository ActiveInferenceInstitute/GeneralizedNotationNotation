# GNN Formal Specifications Module

## Purpose

Static **reference specifications** of GNN-related structures in eight external languages (proof assistants and specification tools). These files are **not** the Python round-trip suite itself; they complement [`../parsers/`](../parsers/) and [`../types.py`](../types.py).

## Files

| File | Language |
|------|----------|
| `isabelle.thy` | Isabelle/HOL |
| `lean.lean` | Lean 4 |
| `coq.v` | Coq |
| `agda.agda` | Agda |
| `alloy.als` | Alloy |
| `z_notation.zed` | Z notation |
| `tla_plus.tla` | TLA+ |
| `maxima.mac` | Maxima |

See **[README.md](README.md)** for focus and how to invoke each tool.

## Relationship to Python parsers

Where the same surface syntax exists in the pipeline (e.g. Lean, Coq, Agda, TLA+, Alloy, Z, Maxima, BNF family), [`../parsers/`](../parsers/) provides **parse/serialize** for interchange. **Isabelle** and other files here are primarily **manual** verification artifacts.

Round-trip **automation** is defined by [`../testing/test_round_trip.py`](../testing/test_round_trip.py) and **[../SPEC.md](../SPEC.md)** — not by “all eight files round-trip at 100%” as a blanket statement.

## For AI agents

1. Treat these as **ground-truth references** for formal semantics, not as guaranteed-identical to every Python serialization path.
2. For runtime format counts and serializer registry facts, use **[../SPEC.md](../SPEC.md)**.
