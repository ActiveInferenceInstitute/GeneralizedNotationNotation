# docs/standards/ — repository standards

## Purpose

Normative standards for this repository: conventions that new code must
follow and that CI gates make load-bearing. A standard is authored from a
census (never invented), cites file:line evidence verified in-tree at
authoring time, and stays in sync with the gates that enforce it.

## Contents

- [composability.md](composability.md) — the five-seam composability
  standard: the five de-facto seams, tracked exceptions, the declared
  cross-repo consumer surfaces, and enforcement.
- [exceptions.md](exceptions.md) — the exception doctrine: when
  `except Exception` is acceptable, the mandatory log + structured-receipt
  obligations, the review checklist, and the occurrence-count ratchet
  (baseline-pinned fail-on-growth probe in
  [scripts/check_gnn_doc_patterns.py](../../scripts/check_gnn_doc_patterns.py)).

## Conventions

- Standards never claim a mechanism that is not landed in the tree.
- Every tracked exception carries a census ref (COMP id), a file:line site,
  and a fix direction; exceptions are never hidden in body text.
- Cross-repo custody wording defers to
  [docs/development/fep_lean_paired_revision.md](../development/fep_lean_paired_revision.md).

## Related

- [docs/decisions/AGENTS.md](../decisions/AGENTS.md) — architecture
  decision records index.
