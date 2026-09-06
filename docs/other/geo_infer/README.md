# geo_infer

> **Document Metadata**
> **Type**: External project documentation (research archive) | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Main documentation](../../README.md)
> **Last Updated**: 2026-09-06 | **Version**: 0.1

## Overview

GEO-INFER is the Active Inference Institute's spatial active-inference
monorepo: 47 modules covering geography, topology, environmental sensing,
and cognitive modeling over spatially indexed data. The sibling checkouts
live at `../GEO-INFER` (paths in this folder that start with `../GEO-INFER/`
are written as inline code on purpose: each repository validates its own
links, so cross-repo references are never markdown links). The integration
is **artifact-level only**: the two projects run in separate environments and
exchange versioned JSON artifacts with SHA-256 provenance; neither checkout
imports the other, and their contract forbids cross-checkout imports.

The relevant modules on the GEO-INFER side are:

- **SPACE** — H3 spatial topology and the state-id space artifacts bind to.
- **TIME** — observation scheduling for spatially distributed observations.
- **ACT** — the inference consumer; its `core.gnn_contract` module is the
  typed consumer boundary for GNN artifacts.
- **TEST** — the cross-repo validation harness that checks GNN-emitted
  artifacts against the contracts.

## Contracts

Three versioned artifact contracts connect the pipeline to GEO-INFER. The
normative format documentation is `src/gnn/export/geo_infer_contract.md`;
the factored variant has its own normative doc,
`src/gnn/export/geo_infer_factored_contract.md`.

| Contract | Producer (this repo) | Content |
| --- | --- | --- |
| `gnn-geo-infer/1` | `src/gnn/export/geo_infer.py` | Categorical models (strict GNN; optional H3 `--space-kind h3` support behind `--extra geo-infer`) |
| `gnn-geo-infer/2` | `src/gnn/export/geo_infer_gaussian.py` | Linear-Gaussian models with `F`/`G`/`H`/`Q`/`R` matrices |
| `gnn-geo-infer/factored/1` | `src/gnn/export/geo_infer_factored.py` | Factored state-space models |

On the consumer side, the typed boundary is
`geo_infer_act.core.gnn_contract`: `GNNArtifact` (the parsed artifact type)
and `run_gnn_inference` (the entry point that drives ACT inference over an
artifact).

## Exporting an artifact

From this checkout:

```
uv run python -m gnn.export.geo_infer input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md /tmp/gridworld.geo-infer.json --step-seconds 60
```

This is the post-reorganization module path (`python -m gnn.export.geo_infer`).
The `--space-kind h3` option requires the optional H3 dependency: install
with `uv sync --extra geo-infer` first.

## Validating interchange

The validation harness lives in the GEO-INFER-TEST module of the sibling
checkout. Run it from the GEO-INFER checkout, pointing it at this repo:

```
cd ../GEO-INFER && uv run python GEO-INFER-TEST/validate_gnn_interchange.py --gnn-repo ../GeneralizedNotationNotation --gnn-python ../GeneralizedNotationNotation/.venv/bin/python
```

The harness re-exports known exemplars with this repo's interpreter and
checks the produced artifacts against the contract schemas and provenance
requirements.

## Relationship to fep_lean

fep_lean verifies the well-formedness of GNN **source documents** in Lean 4,
upstream of export (see [docs/other/fep_lean/](../fep_lean/README.md));
GEO-INFER consumes the **exported artifacts** downstream. No Lean toolchain
is involved anywhere in the GEO-INFER interchange.

## Related documentation

- [fep_lean bridge program](../fep_lean/README.md) — upstream document
  well-formedness verification
- [Framework integration guide](../../gnn/integration/framework_integration_guide.md)
- [docs/other archive](../README.md) — this folder's parent

The canonical interchange documentation lives in the sibling checkout at
`../GEO-INFER/GEO-INFER-ACT/docs/gnn_interchange.md`.
