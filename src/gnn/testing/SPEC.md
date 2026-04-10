# Testing — specification

## Role

- **`test_round_trip.py`** — `GNNRoundTripTester`; round-trip harness (default **21** format strings in config; see **[../SPEC.md](../SPEC.md)**).
- **`test_comprehensive.py`**, **`test_integration.py`**, **`test_xml_parser_only.py`**, **`simple_round_trip_test.py`** — focused tests.
- **`performance_benchmarks.py`**, **`round_trip_strategy.py`** — benchmarks / helpers.
- **`round_trip_reports/`** — optional output directory for reports.

## Requirements

- **Python** >= 3.11 (see repo `pyproject.toml`).

## Running

```bash
uv run pytest src/gnn/testing/ -q
uv run pytest src/tests/test_gnn*.py -q
```

See **[README.md](README.md)** and **[README_round_trip.md](README_round_trip.md)**.
