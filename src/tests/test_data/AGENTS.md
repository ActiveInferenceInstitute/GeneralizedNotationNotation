# Test Data — Agent Scaffolding

## Purpose

Provide deterministic GNN fixtures that tests can consume without reaching into
`input/gnn_files/`, so the test suite stays stable even when sample models in
`input/` are reorganised.

## Fixture Inventory

| File | Sections | Shape | Used By |
|------|----------|-------|---------|
| `sample_gnn_model.md` | `StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`, `Time`, `ActInfOntologyAnnotation`, `ModelParameters` | A[3,3], B[3,3,3], C[3], D[3], E[3], s[3,1], o[3,1] | Parser, type-checker, export round-trip, render backends, visualization tests |

## Invariants

- **Read-only**: tests must not write into this directory. Use `tmp_path` for
  scratch files.
- **Stable content**: changes to `sample_gnn_model.md` cascade into many tests.
  Coordinate updates with `src/gnn/`, `src/type_checker/`, and
  `src/render/` maintainers.
- **ASCII + Markdown**: fixtures stay text-only so diffs are reviewable.

## Related Files

- `src/tests/conftest.py` — fixtures that resolve paths relative to this directory
- `doc/gnn/gnn_examples_doc.md` — the reference model documentation
- `input/gnn_files/` — full-scale models (not fixtures)

## Documentation

- **[README](README.md)** — overview
- **[AGENTS](AGENTS.md)** — this file
