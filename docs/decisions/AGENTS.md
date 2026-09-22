# docs/decisions/ — Architecture Decision Records

## Purpose

Append-only ADRs. Numbered `NNNN-<kebab-title>.md`. Never edit an
accepted ADR's decision; supersede it with a new record that links
backward. Every ADR carries **Status** (Accepted/Superseded/Deferred),
**Date**, and the SCOPE/issue item that motivated it.

Current records:

- `0001-consolidated-pipeline-execution.md` — opt-in in-process step
  executor behind `--consolidated-steps`; numbered-script contract
  unchanged; first slice covers steps 0/3/5 on flat target dirs
  (SCOPE-2026-09-10 / S2-11).
- `048-composition-seam.md` — supersedes the imported tool-level
  composition surface (`compose`/`pipe`/`find_tools`/`lift`, never landed
  here) with ADR 0001's registry-driven step executor plus the five-seam
  doctrine (`docs/standards/composability.md`); the tool surface stays
  unadopted absent a consumer-demand receipt (five-seam census /
  COMP-001, COMP-002).

New ADRs must be listed here.
