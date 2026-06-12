# GNN Folder Alignment Status

**Last updated:** 2026-04-12

**Reference model:** `input/gnn_files/actinf_pomdp_agent.md` (also under `src/gnn/gnn_examples/`)

**Purpose:** Track alignment of `src/gnn/` with the reference GNN model: schemas/grammars, parsers, validators, documentation, and round-trip behavior.

Canonical **format counts** (23 enum, 22 serializers, round-trip scope): see **[SPEC.md](SPEC.md)**.

**File discovery:** `processor.discover_gnn_files()` uses narrow globs (`*.md`, `*.gnn`, `*.txt`); Step 3 `process_gnn_multi_format()` uses a broad extension list — see SPEC § File discovery.

## Round-trip testing (default suite)

The suite in [`testing/test_round_trip.py`](testing/test_round_trip.py) uses **21** format strings in `FORMAT_TEST_CONFIG['test_formats']`: **`markdown`** plus **20** conversion targets. For the reference model, that suite reports **100%** success.

**Not in the default round-trip list:** **`ebnf`** (same `GrammarSerializer` path as **BNF**; not separately exercised), **`pnml`** (disabled in config; PNML remains **parse**-focused in `parsers/system.py`).

**Serializers:** `SERIALIZER_REGISTRY` in [`parsers/system.py`](parsers/system.py) has **22** entries (no PNML serializer). **`GNNFormat`** has **23** values.

### Schema formats (7/7 in suite)

JSON, XML, YAML, Protobuf, XSD, ASN.1, PKL — round-trip with embedded model data where applicable.

### Language formats (6/6)

Python, Scala, Lean, Coq, Isabelle, Haskell.

### Formal / grammar (5 + BNF; EBNF noted above)

TLA+, Agda, Alloy, Z-notation, BNF.

### Other (2/2)

Maxima, Pickle.

### Reference

Markdown (source format for the reference file).

## Embedded model data

Serializers embed a JSON snapshot of the model in comments or equivalent so parsers can restore semantics after `serialize → parse`. Patterns vary by format (e.g. `MODEL_DATA` in comments). Implementation is spread across [`parsers/*_serializer.py`](parsers/) modules, coordinated by [`parsers/system.py`](parsers/system.py).

## Folder structure (high level)

| Path | Role |
|------|------|
| `parsers/system.py` | `PARSER_REGISTRY`, `SERIALIZER_REGISTRY`, `GNNParsingSystem` |
| `parsers/common.py` | `GNNFormat`, protocols, shared types |
| `parsers/*_parser.py`, `*_serializer.py` | Per-format I/O |
| `schemas/` | JSON, YAML, XSD, Proto, ASN.1, PKL definitions |
| `grammars/` | BNF / EBNF grammars |
| `testing/test_round_trip.py` | Round-trip harness |
| `gnn_examples/` | Reference Markdown models |
| `formal_specs/` | Standalone formal artifacts (Isabelle, Lean, Coq, etc.) |

## Technical notes

- **Parsers:** 23 registered in `PARSER_REGISTRY`.
- **Serializers:** 22 registered; PNML is not serializer-registered (see SPEC.md).
- **Cross-format validation:** `cross_format_validator.py`; optional in tests via config flags.
- **Unicode:** Policy variables such as π supported in grammars and reference models.

## History (January 2025 milestone)

Round-trip reliability for the then-configured format set was brought to **100%** for the reference model, with embedded-data serialization and parser fixes documented in `testing/README_round_trip.md`.

---

**Summary:** Alignment is measured against the reference POMDP-style model and the **default** round-trip configuration. For enum vs serializer vs test scope, always use **[SPEC.md](SPEC.md)**.
