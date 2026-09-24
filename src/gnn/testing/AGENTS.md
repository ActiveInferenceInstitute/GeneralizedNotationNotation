# Testing — agent notes

## Purpose

GNN-local helpers and benchmarks: round-trip (`round_trip_tester.py` + `round_trip_*` siblings), integration, XML-only paths, and performance scripts. pytest cases live under `tests/testing/`.

## Canonical numbers

Round-trip **scope** vs **enum size** is documented in **[../SPEC.md](../SPEC.md)**. Do not duplicate conflicting format counts here—link to SPEC.

## Key files

- **`round_trip_tester.py`** — primary round-trip harness (`GNNRoundTripTester`); tune `FORMAT_TEST_CONFIG` in `round_trip_config.py`.
- **`README_round_trip.md`** — methodology.
- **`alignment_status.md`** — alignment snapshot (also references SPEC).

## Commands

```bash
uv run --extra dev python -m pytest tests/testing/ -q
uv run --extra dev python -m pytest tests/gnn/test_gnn_overall.py -q
```
