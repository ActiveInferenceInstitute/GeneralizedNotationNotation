# Testing — agent notes

## Purpose

GNN-local tests and benchmarks: round-trip (`test_round_trip.py`), integration, XML-only paths, and performance scripts.

## Canonical numbers

Round-trip **scope** vs **enum size** is documented in **[../SPEC.md](../SPEC.md)**. Do not duplicate conflicting format counts here—link to SPEC.

## Key files

- **`test_round_trip.py`** — primary round-trip harness; tune `FORMAT_TEST_CONFIG` at top of file.
- **`README_round_trip.md`** — methodology.
- **`../alignment_status.md`** — alignment snapshot (also references SPEC).

## Commands

```bash
uv run pytest src/gnn/testing/test_round_trip.py -q
uv run pytest src/tests/test_gnn_overall.py -q
```
