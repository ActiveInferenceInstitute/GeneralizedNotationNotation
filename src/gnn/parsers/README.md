# GNN parsers

Multi-format **parse** and **serialize** for GNN models: one `*_parser.py` / `*_serializer.py` pair per format family, plus shared infrastructure.

## Entry points

- **`GNNParsingSystem`** in `system.py` — `parse_file`, `serialize`, format conversion.
- **`PARSER_REGISTRY`** / **`SERIALIZER_REGISTRY`** — map `GNNFormat` to concrete classes.
- **`GNNFormat`**, **`GNNParser`** (protocol) — `common.py`.

Canonical **enum size**, registry counts, and round-trip scope: **[../SPEC.md](../SPEC.md)**.

## Layout

See **[SPEC.md](SPEC.md)** for the full layout table (`grammar_*`, `schema_*`, `xml_*`, `unified_parser.py`, `validators.py`, etc.).

## Connection annotations

`Connection.annotation` preserves the label after a connection's colon without changing source or target names. JSON/YAML interchange and embedded model dictionaries serialize this optional field, and reconstruction restores it. Older dictionaries that omit it retain `None`. The focused regression `src/tests/gnn/test_connection_annotation_roundtrip.py` exercises absent and Unicode labels across every registered serializer/parser pair; this is label-fidelity evidence, not a claim of complete model fidelity for every format.

## Pickle inputs

Pickle inputs use `safe_pickle_load` and `safe_pickle_loads` in
`binary_parser.py`. Both inspect the same bytes they reconstruct, reject
extension opcodes that could bypass the global allowlist, and reject bytes
after the first record. Parsing, schema validation and export validation use
this shared boundary. It limits object reconstruction; it does not authenticate
artifacts or bound memory and CPU use.

## Adding a format

1. Extend **`GNNFormat`** in `common.py` if needed.
2. Implement parser and (unless parse-only) serializer classes.
3. Register in **`PARSER_REGISTRY`** and **`SERIALIZER_REGISTRY`** in **`system.py`**.
4. Add tests under `src/tests/` and extend `src/gnn/testing/test_round_trip.py` if the format should join the default round-trip list.

## Tests

```bash
uv run --extra dev python -m pytest src/tests/gnn/test_gnn_parsing.py src/tests/gnn/test_gnn_parsers_common.py -q
```

Agent-oriented detail: **[AGENTS.md](AGENTS.md)**.
