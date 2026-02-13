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
4. **Round-trip fidelity**: Examples must round-trip through all supported formats with semantic preservation

## Testing

Examples are used as input fixtures by:

- `gnn/testing/test_round_trip.py` — round-trip across all 23 formats
- `gnn/testing/test_comprehensive.py` — comprehensive parsing validation
- `src/tests/test_gnn_parsing.py` — unit-level parsing tests
