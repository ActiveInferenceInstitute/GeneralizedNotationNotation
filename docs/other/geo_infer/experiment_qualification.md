# Cascadia experiment export qualification

> **Document Metadata**
> **Type**: Operational guide | **Audience**: Researchers, Developers
> **Cross-References**: [README.md](README.md) | [cascadia_river_homing.md](cascadia_river_homing.md)
> **Last Updated**: 2026-10-13 | **Version**: 0.1

## A separate export for each assumption

A GEO-INFER study retains one GNN source, ordered H3 identifier array, and
artifact per condition. [qualify_export.py](qualify_export.py) calls the real
`gnn.export.geo_infer.build_geo_infer_artifact` producer again and compares
its complete result with the retained artifact. This is a read-only producer
reproduction check; it neither reruns the simulation nor compiles Lean.

```bash
uv run python docs/other/geo_infer/qualify_export.py \
  ../GEO_INFER/GEO-INFER-ACT/examples/outputs/STUDY/uniform_habit \
  --step-seconds 21600 --output output/uniform_habit-export-check.json
```

The directory must contain `smith_river_homing.gnn.md`,
`smith_river_homing.state_ids.json`, and `smith_river_homing.geo-infer.json`.
The clock argument is explicit and must match the intended experiment.
Inputs use the producer's four-MiB per-file limit. Duplicate JSON keys and
non-finite numbers are rejected. Canonical JSON comparison preserves the
distinction between numbers and booleans (`1.0` is not `true`). The optional
output is create-only, with an existing parent;
omitting it prints the qualification result.

The result records exact source, identifier-file, and artifact digests plus
the producer revision, qualifier digest and current producer-source roster
digests. The roster includes Python owners, dependency files, notation syntax
and the bridge contract; it is checked before and after reproduction.
`export_reproduced: true` means
the current producer reconstructed that artifact. `native_claim_ready` is
always false. Revision identity alone does not establish that a checkout is
clean; preserve the study's own source-owner record and inspect local changes
when using an execution receipt. The schema is
`gnn/cascadia-export-qualification/2`. Earlier receipts remain historical;
they cannot establish the new roster identity.

Use `--check-receipt PATH` instead of `--output PATH` to reproduce the export
and compare a saved receipt with the current qualifier and checkout bytes.
It writes nothing and fails on any difference. These digests identify source
files at qualification, not the complete installed dependency environment or
the state of a previously imported module in a long-running interpreter.

## Roles of the adjacent checks

| Check | What it establishes |
| --- | --- |
| GNN export reproduction | Source, ordered identifiers and explicit clock reproduce the artifact |
| GEO study validation | Retained hashes, ordered grid, outcomes, summaries and run metadata agree |
| GEO closed-loop execution | Sampled observations replay through the public inference contract |
| fep_formal proof-impact worksheet | Changed exact tables/carrier identify reopened proof obligations |
| Named native Lean receipt | Only the exact compiled statements bound to its inputs |
| LectureCreate companion receipt | The named offline companion and subprocess outputs were produced |

LectureCreate's `examples/cascadia_geo_act/study_companion.py` orchestrates
the first, second and fourth checks and the interactive visual. Each tool
runs with its own checkout's `uv` environment. It imports no sibling source
and makes no publication or video-render claim.

## Tests and failure triage

```bash
uv run python -m pytest docs/other/geo_infer/test_qualify_export.py -q --no-cov
uv run python -m pytest tests/export/test_geo_infer_cascadia.py -q --no-cov
uv run ruff check docs/other/geo_infer/qualify_export.py docs/other/geo_infer/test_qualify_export.py
```

Tests reconstruct the golden reference and reject an altered habit prior,
source digest, H3 order, clock, duplicate key and boolean/numeric alias. Receipt
rechecks reject altered producer digests. A second write must fail
while preserving the first receipt. Fix the source or metadata when export
fails; never normalize tables or repair a retained artifact inside the checker.
