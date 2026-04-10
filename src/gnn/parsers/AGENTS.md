# Parsers — agent notes

## Purpose

Multi-format I/O for GNN: **23** `GNNFormat` values, **23** parsers, **22** serializers in **`system.py`** (see **[../SPEC.md](../SPEC.md)**).

## Entry points

- **`GNNParsingSystem`** — instantiate and call `parse_file`, `serialize`, etc.
- **`PARSER_REGISTRY` / `SERIALIZER_REGISTRY`** — maps `GNNFormat` to concrete classes.
- **`GNNFormat`**, **`GNNParser`** (protocol) — **`common.py`**.

## Adding a format

1. Extend **`GNNFormat`** in `common.py` if needed.
2. Implement parser (and serializer unless parse-only) classes.
3. Register in **`PARSER_REGISTRY`** and, if applicable, **`SERIALIZER_REGISTRY`** in **`system.py`**.
4. Add tests under `src/tests/` and extend `src/gnn/testing/test_round_trip.py` if the format should participate in round-trip.

## Tests

```bash
uv run pytest src/tests/test_gnn_parsing.py src/tests/test_gnn_parsers_common.py -q
```
