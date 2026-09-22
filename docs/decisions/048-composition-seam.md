# ADR 048: Composition seam

- **Status:** Accepted
- **Date:** 2026-09-21
- **Scope item:** five-seam census, 2026-09-21 (COMP-001, COMP-002)
- **Companion standard:** [docs/standards/composability.md](../standards/composability.md)
- **Links backward:** [ADR 0001 — Consolidated pipeline execution](0001-consolidated-pipeline-execution.md)

## Context

**Provenance.** The five-seam doctrine — orchestrator→module processor,
pipeline.config output-dir/registry, the `gnn.utils` PEP 562 facade, kernel
packages, registries — originates in the parent platform monorepo, where it is paired
with a tool-level composition surface (`compose`, `pipe`, `find_tools`,
`lift`) and its own composition-seam record. GNN imported the doctrine as an
aspirational standard; until this record, neither the doctrine nor the tool
surface existed on main (the census receipt: no standards doc, no ADR for
it, no code, no tests, no callers).

**Census receipts, re-verified in this tree:**

- The tool-level surface was never landed in GNN: zero Python definitions
  of `compose`/`pipe`/`find_tools`/`lift` across `src/`, `tests/`, and
  `scripts/` (the only textual hits are an unrelated Docker-compose plan
  renderer — `plan_to_compose` at
  `src/gnn/pipeline/container_plan.py:388` — and third-party prose under
  `docs/other/`).
- GNN's landed canonical composition mechanism is ADR 0001's
  registry-driven step executor: `execute_step_in_process`
  (`src/gnn/pipeline/step_executor.py`), the whitelist
  `CONSOLIDATED_IN_PROCESS_STEMS`
  (`src/gnn/pipeline/step_registry.py:355`), dispatched through
  `_execute_selected_step` (`src/gnn/main.py:977`) in both the serial loop
  and the parallel tier.
- The five de-facto seams are real and, after this wave's remediations,
  conformant or exception-tracked; they are codified in
  [docs/standards/composability.md](../standards/composability.md).

**Weighing land vs supersede:**

- **Land case.** The doctrine already exists in parent workspace, so landing the
  functions would keep one mental model across the repository family; a
  shared composition vocabulary could eventually serve MCP tool
  composition and cross-repo reuse.
- **Supersede case.** There is zero in-repo demand today — no consumer, no
  caller, no test (the census receipts above). A landed mechanism already
  covers step composition: ADR 0001's executor, with its single reviewable
  whitelist seam. Module-level composition is already covered by the five
  seams. A parallel mechanism for the same job is a second convention:
  duplicate maintenance, two vocabularies to keep honest, and drift risk —
  precisely the failure mode the five-seam standard exists to prevent.
  Landing code nobody consumes is speculative surface area.

## Decision

GNN **supersedes** the imported tool-level composition surface for this
repository. The canonical composition mechanisms are:

1. the five de-facto seams, codified in
   [docs/standards/composability.md](../standards/composability.md);
2. the registries as the single mapping per family; and
3. ADR 0001's registry-driven consolidated step executor (whitelist
   `CONSOLIDATED_IN_PROCESS_STEMS`) as the single seam for in-process step
   composition.

`compose`, `pipe`, `find_tools`, and `lift` are **not adopted**. They
remain unadopted absent a consumer-demand receipt — a concrete consumer
that needs function-level composition across modules and cannot be served
by the executor and the seams. Any future adoption supersedes this record
and must land implementation, tests, and documentation in one slice.

## Consequences

**Positive**

- One composition story: every contributor reads the five-seam standard and
  ADR 0001 instead of choosing between two mechanisms.
- No speculative surface: nothing is landed without a consumer.
- The parent-monorepo doctrine import is preserved where it is true here (the seams)
  and explicitly delimited where it is not (the tool surface).

**Negative / accepted limitations**

- Cross-repo doctrine continuity is carried by documentation, not shared
  code: a contributor arriving from parent workspace must be told that the tool
  surface is not part of GNN.
- If a genuine consumer appears later, building the tool surface then costs
  more than building it now would have. Accepted: the census shows no such
  consumer today, and the seams cover the composition needs the codebase
  actually expresses.

## References

- [docs/standards/composability.md](../standards/composability.md) — the
  five seams, tracked exceptions, consumer surfaces, enforcement
- [ADR 0001 — Consolidated pipeline execution](0001-consolidated-pipeline-execution.md)
  — the landed executor and `CONSOLIDATED_IN_PROCESS_STEMS`
- `src/gnn/pipeline/step_executor.py`;
  `src/gnn/pipeline/step_registry.py:355`; `src/gnn/main.py:977`
- Five-seam census, 2026-09-21 (COMP-001/COMP-002 receipts; exception
  tracker COMP-003 through COMP-011)

## Status log

| Date | Slice | Status | Notes |
|------|-------|--------|-------|
| 2026-09-21 | Initial decision | Accepted | Supersede recorded; the standard authored in the same wave; import-linter contracts wired and green (3 kept, 0 broken) |
