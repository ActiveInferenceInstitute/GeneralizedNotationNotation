# geo_infer (docs/other archive entry)

## Overview

**Status**: active documentation program (draft) | **Version**: 0.1 | **Last
Updated**: 2026-10-12

## Purpose

Document, for GNN-side readers and agents, the artifact-level interchange
with the sibling spatial active-inference monorepo `GEO_INFER`: how GNN
models are exported as versioned JSON artifacts, and how GEO_INFER consumes
and validates them. This folder is the GNN-side mirror of the canonical
interchange documentation at `../GEO_INFER/GEO-INFER-ACT/docs/gnn_interchange.md`
(cross-repo references are inline code paths, never markdown links). The
sibling is a foreign git repository: never commit into it from GNN tooling.

## Contents

| File | Role |
| --- | --- |
| [README.md](README.md) | Reader-facing entry point: contracts, export command, validation harness |
| [cascadia_river_homing.md](cascadia_river_homing.md) | Worked H3 example: the Cascadia GNN source, exporter checks and fixtures |
| [california_monitoring.md](california_monitoring.md) | Factored indicator sensors, producer qualification and statewide consumer boundaries |

## Key paths (post-reorg)

For separate agent/process experiments, read [model_pairs.md](model_pairs.md).
Reproduce both exports independently and keep semantic index alignment distinct
from equal dimensions. Consumer execution changes belong to GEO-INFER-ACT.
For configured registries, follow [model_matrices.md](model_matrices.md) and
qualify every selected source before attributing a matrix input to its notation.

- Producers: `src/gnn/export/geo_infer.py` (categorical),
  `src/gnn/export/geo_infer_gaussian.py` (Gaussian),
  `src/gnn/export/geo_infer_factored.py` (factored).
- Normative format docs: `src/gnn/export/geo_infer_contract.md` and
  `src/gnn/export/geo_infer_factored_contract.md`.
- Export entry point: `uv run python -m gnn.export.geo_infer ...`.
- Consumer boundary (GEO_INFER side): `geo_infer_act.core.gnn_contract`
  (`GNNArtifact`, `run_gnn_inference`).
- Validation harness (GEO_INFER side):
  `GEO-INFER-TEST/validate_gnn_interchange.py`.

## Version facts

- Contracts: `gnn-geo-infer/1` (categorical), `gnn-geo-infer/2` (Gaussian
  `F`/`G`/`H`/`Q`/`R`), `gnn-geo-infer/factored/1` (factored).
- Integration is artifact-level only: separate environments, JSON artifacts
  with SHA-256 provenance, no cross-checkout imports (the GEO_INFER contract
  forbids it).
- H3 support is optional (`--extra geo-infer`).
- The Cascadia seed study emits variant artifacts into its own output edition;
  retain source/state order/export custody and never promote reference golden
  fixtures or native Lean receipts to variant claims.

## Editing rules

- `qualify_factored.py` invokes the actual factored producer. Preserve bounded
  source/artifact reads, distinct JSON fields and create-only generation.
  Reproduction never repairs drift. Run
  `uv run pytest docs/other/geo_infer/test_qualify_factored.py tests/export/test_geo_infer_factored.py -q --no-cov`.

- `qualify_export.py` is a thin worked-example client of the existing producer.
  Keep it read-only on inputs and create-only on optional receipts. Run
  `uv run python -m pytest docs/other/geo_infer/test_qualify_export.py -q --no-cov`
  after changing it. Its reproduction result never implies native Lean proof.
  New receipts bind producer-source bytes and use schema
  `gnn/cascadia-export-qualification/2`; `--check-receipt` is read-only and
  fails on qualifier, source-roster or artifact drift. Boolean/numeric aliases
  must not pass canonical artifact comparison.

- Keep all links relative and verify targets exist; the repository-wide
  documentation audit (`uv run python
  docs/development/docs_audit.py --strict --check-anchors --no-write`) checks
  them for every markdown file.
- Follow [style_guide.md](../../style_guide.md): `uv run python` command
  spellings, H1 title, metadata block, honest claims without embedded
  ungenerated counts.
- Update this folder together with the producer modules under
  `src/gnn/export/`; contract changes bump the contract versions named here.

---

**Status**: active | **Maintenance**: update together with the export
modules under `src/gnn/export/` | **Last Updated**: 2026-10-12
