# GNN Examples — Specification

## Architecture

This module provides **reference GNN model files** in Markdown format. These serve a dual role as documentation and as test fixtures for the round-trip validation system.

## Components

| Component | Type | Purpose |
|-----------|------|---------|
| `actinf_pomdp_agent.md` | Reference Model | Canonical Active Inference POMDP agent specification demonstrating all GNN sections |

## Requirements

1. **Structural completeness**: Example files must include all required GNN sections (`ModelName`, `StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`, `Time`)
2. **Syntactic correctness**: All GNN punctuation and notation must conform to `documentation/punctuation.md`
3. **Parse-ability**: Each example must parse successfully through `gnn/parsers/markdown_parser.py`
4. **Round-trip fidelity**: Reference models should round-trip for every format in the **default** `test_formats` list in `gnn/testing/test_round_trip.py` (see **[../SPEC.md](../SPEC.md)**). That list is **not** identical to all **23** `GNNFormat` values (e.g. **PNML** / **EBNF** may be excluded or shared with BNF machinery).

## Testing

Examples are used as input fixtures by:

- `gnn/testing/test_round_trip.py` — round-trip for configured formats (see **[../SPEC.md](../SPEC.md)**)
- `gnn/testing/test_comprehensive.py` — comprehensive parsing validation
- `src/tests/test_gnn_parsing.py` — unit-level parsing tests
