# Testing — specification

## Role

- **`round_trip_tester.py`** — `GNNRoundTripTester`; round-trip harness (default **21** format strings in config; see **[../SPEC.md](../SPEC.md)**). Pytest coverage lives under `tests/testing/`.
- **`round_trip_config.py`**, **`round_trip_results.py`**, **`round_trip_comparison.py`**, **`round_trip_report.py`**, **`round_trip_markdown_parser.py`**, **`round_trip_availability.py`**, **`round_trip_strategy.py`** — harness configuration / results / mixins / availability probe / strategy.
- **`performance_benchmarks.py`** — benchmarks helper (exercised by `tests/testing/`).
- **`round_trip_reports/`** — optional output directory for reports.

## Requirements

- **Python** >= 3.11 (see repo `pyproject.toml`).

## Running

```bash
uv run --extra dev python -m pytest tests/testing/ -q
uv run --extra dev python -m pytest tests/test_gnn*.py -q
```

See **[README.md](README.md)** and **[README_round_trip.md](README_round_trip.md)**.
