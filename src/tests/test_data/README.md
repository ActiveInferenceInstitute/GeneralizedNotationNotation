# Test Data

Static GNN fixtures used by the test suite. Files here are treated as read-only
inputs; tests that need scratch space should use pytest's `tmp_path` fixture.

## Contents

| File | Purpose |
|------|---------|
| `sample_gnn_model.md` | Canonical Active Inference POMDP model (3 observations × 3 hidden states × 3 actions) exercised by parser, type-checker, exporter, renderer, and round-trip tests |

The sample model is the same ActInf POMDP used by `doc/gnn/gnn_examples_doc.md` and
the walkthroughs in `doc/gnn/reference/`, so it is a stable reference point for
cross-module assertions.

## Adding Fixtures

1. Keep fixtures small — full-scale models belong in `input/gnn_files/`, not here.
2. Prefer reusing `sample_gnn_model.md` over adding near-duplicates.
3. Add a short description to the table above when a new fixture is introduced.
4. Fixtures must be deterministic — no timestamps, UUIDs, or randomised content.

## Documentation

- **[README](README.md)** — this file
- **[AGENTS](AGENTS.md)** — agent-facing overview
