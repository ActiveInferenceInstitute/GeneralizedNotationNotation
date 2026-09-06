# Extract Module - Agent Scaffolding

## Module Overview

**Purpose**: Headless extraction of POMDP state spaces from GNN specification files behind a versioned JSON envelope contract

**Pipeline Step**: Infrastructure module — invoked by the `extract` handler in the CLI (`src/gnn/cli/__init__.py` wires `extract_to_json`); not a numbered pipeline step

**Category**: Extraction / Headless entry point

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-06

---

## Module Structure

```
src/gnn/extract/
├── __init__.py            # JSON envelope facade: extract_to_json, main (schema 1.0.0)
├── pomdp_extractor.py     # POMDP/state-space extraction (stdlib-only at module scope)
├── __main__.py            # python -m gnn.extract FILE [options] entry point
├── README.md              # Module documentation
└── AGENTS.md              # This agent scaffolding documentation
```

---

## Core Functionality

### Primary Responsibilities
1. Extract POMDP state spaces (factors, modalities, matrices, provenance) from GNN files.
2. Wrap extraction in a versioned, never-raising JSON envelope (schema 1.0.0).
3. Provide the documented headless CLI entry: `python -m gnn.extract FILE [--strict|--no-strict] [--compact]`.

### Key Capabilities
- `extract_to_json(path, *, strict_validation=True, on_error="lenient", compact=False) -> str` — always returns JSON; success payload is `POMDPStateSpace.to_dict()`, failure is the `{"status": "error", "error": {...}}` envelope.
- `main(argv=None) -> int` — CLI wrapper; exit 0 on success, exit 1 on failure (repo convention).
- `POMDPExtractor` — stdlib-only parser producing `POMDPStateSpace` (state factors, observation modalities, control factors, A/B/C/D/E matrices with `matrix_provenance`, continuous families with `x, F, H, Q, R`).
- `extract_pomdp_from_file` / `extract_pomdp_from_content` / `canonicalize_pomdp` — functional surface of the extractor module.
- `OnErrorMode` (`"lenient" | "raise" | "collect"`) — structured error surfacing; parameter-parse failures are recorded in `matrix_provenance` in every mode.

---

## Cross-Repo Pin (do not move)

The fep_lean bridge `verify-document` operation imports `gnn.extract.pomdp_extractor.extract_pomdp_from_file` as its render route (bridge contract `doc/other/fep_lean/bridge-contract.md` §13, mirrored in the fep_lean checkout). This import path is part of the cross-repo source-custody pin: any move or rename must be coordinated with the fep_lean checkout and a re-pin of `specs/gnn-bridge-w2-source-custody/source-pin.json`.

`__main__.py` preserves `python -m gnn.extract` as the documented headless entry; `__init__.py` imports only the standard library at module scope so the CLI runs without the full pipeline stack.

---

## Agent Instructions

- The public surface is exactly `__all__ = ["extract_to_json", "main"]` in `__init__.py`; the extractor module's functions are importable by path.
- Keep `pomdp_extractor.py` stdlib-only at module scope; heavy or pipeline imports must stay lazy inside call paths.
- The JSON envelope schema version is `1.0.0`; additive keys only, no breaking changes to `status`/`error` shapes.
- Tests: `tests/cli/test_cli_extract.py`, `tests/gnn/test_pomdp_extractor_continuous.py`, `tests/gnn/test_pomdp_extractor_counts.py`, `tests/gnn/test_pomdp_extractor_errors.py`, `tests/gnn/test_pomdp_extractor_isolation.py`, `tests/gnn/test_pomdp_extractor_orientation.py`.
- Maintain `README.md` alongside any surface change; the per-directory doc audit checks it.
