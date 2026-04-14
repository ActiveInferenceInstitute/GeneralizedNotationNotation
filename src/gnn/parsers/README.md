# GNN parsers

Multi-format **parse** and **serialize** for GNN models: one `*_parser.py` / `*_serializer.py` pair per format family, plus shared infrastructure.

## Entry points

- **`GNNParsingSystem`** in `system.py` — `parse_file`, `serialize`, format conversion.
- **`PARSER_REGISTRY`** / **`SERIALIZER_REGISTRY`** — map `GNNFormat` to concrete classes.
- **`GNNFormat`**, **`GNNParser`** (protocol) — `common.py`.

Canonical **enum size**, registry counts, and round-trip scope: **[../SPEC.md](../SPEC.md)**.

## Layout

See **[SPEC.md](SPEC.md)** for the full layout table (`grammar_*`, `schema_*`, `xml_*`, `unified_parser.py`, `validators.py`, etc.).

## Adding a format

1. Extend **`GNNFormat`** in `common.py` if needed.
2. Implement parser and (unless parse-only) serializer classes.
3. Register in **`PARSER_REGISTRY`** and **`SERIALIZER_REGISTRY`** in **`system.py`**.
4. Add tests under `src/tests/` and extend `src/gnn/testing/test_round_trip.py` if the format should join the default round-trip list.

## Tests

```bash
uv run pytest src/tests/test_gnn_parsing.py src/tests/test_gnn_parsers_common.py -q
```

Agent-oriented detail: **[AGENTS.md](AGENTS.md)**.
