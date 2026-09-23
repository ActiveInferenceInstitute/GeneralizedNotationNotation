# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-23 (full reconcile: closed-work prose purged per
the Conventions — audit trail lives in `CHANGELOG.md` and git history;
batch-7 PRs #165-#171, the double fep_formal re-seal `5de75c6`→`3aabf31`
with pair bump `966a5b196`, and the custody-re-render preamble fix
`cbee91e02` all landed and swept green. Dispositions verified at
`cbee91e02`: BC-01 RESOLVED (ADR 048 = SUPERSEDE), BC-16 LANDED, SC-38
tail LANDED, BC-15 verified-clean (`src/gnn/lsp/SPEC.md:12` parity claim
accurate — no paragraph needed). Full program: `SCOPE-2026-09-23.md`.)
**Current Version**: 3.5.0
**Next Target**: v4.0.0 (bounded autonomy, pipeline stage consolidation,
multi-agent stigmergic topologies, high-dimensional active inference)

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container
run, or cluster mutation is automatic. The concrete v4.0.0 work is the
scoped program below plus `SCOPE-2026-09-23.md`; this section records the
unscoped vision and the current proposal-only surface.


## Open scoped work

Every open item is cold-startable: scope, files, verification, and
acceptance pinned in `SCOPE-2026-09-23.md` (the authoritative program
file). Summary:

| ID | Class | Scope | Acceptance anchor |
| --- | --- | --- | --- |
| Minor batch (MI-1..MI-16) | minor | 16 × S: stale 3.4.0/3.3.0 stamps (VERSION_MAP, .agent_rules, OPTIONAL_DEPENDENCIES, README bullet, CHANGELOG link defs, docs/gnn corpus), dead API-reference paths (MI-7), dead sapf files (MI-9/10), pip→uv message, test-stub/fixture dedup, shadow-file doc cites | SCOPE-2026-09-23 §Minor |
| BC-02a (+BC-02c, ARCH-1 orphan test) | major | Collapse `validate_gnn_syntax` regex dual path → formal parser delegation; silent-fallback triad; adopt `src/gnn/testing/test_round_trip.py` into tests/ | parsers/basic.py:271-345 vs schema_validator; SCOPE §Major |
| BC-02b | medium | Rename the second live `GNNValidator` (parsers/validators.py:97) | one class + parity tests |
| BC-13 (M-02) | medium | V4-STAGE in-process force-kill: subprocess-worker or cooperative CancelToken; `force_killed` receipt field; parity tests | pipeline/step_executor.py:536-577; ADR 0001 |
| BC-14 | minor | `utils/pipeline_orchestration/pipeline_template.py` I5 fallback chain — conditional on lint-imports going red (currently 3 kept / 0 broken); M-06 receipts work is the unconditional core | SCOPE §Medium M-06 |
| M-01 | medium | Oversized band: 10 files >1200 lines (main.py 1993 … render/processor >1283); decompose one file per wave, parity-gated | SCOPE §Medium M-01 |
| GEN-1..GEN-4 | major | v4.0.0 generalization: continuous executor contract; composed kind detection; factored/hybrid/multi-agent continuous; non-stationary F_t/regime semantics | SCOPE §Major |
| M-03/M-04/M-05/M-07/M-08 | medium | Silent-fallback hotspots (syntax.py:268 tuple, website/renderer ×7, pymdp VFE 0.0, envelope elapsed) | SCOPE §Medium |
| M-09/M-10/M-11/M-12/M-14 | medium | Test-structure + execution-route normalization (7 unmirrored dirs, src-shipped tests, lean CancelToken, MCP pymdp gate bypass, bnlearn seam) | SCOPE §Medium |
| ARCH-3 | minor | Promote exact MCP tool-count pin (162/36) from audit_report.json | tests/mcp/test_registry_internals.py |


Cross-repo (fep_lean coordinator territory, NOT GNN waves): X-1
owner-roster shrink; X-2 one-transaction `repin` bridge operation.

Standing discipline: paired-repin — the fep_formal source-pin seals GNN
owner digests at pin time; ANY later owner-file edit re-drifts the pair.
Ordering: all content edits → token ritual → bridge re-pin → fep PR/merge
→ pair-pin bump as the FINAL commit, single push
(`docs/development/fep_lean_paired_revision.md`). Pre-seal gate battery
(terminology strict, flag-parity, docs audits, token strict) runs BEFORE
pin+emit so a gate failure can never force a post-seal src edit.
`--gnn-root` for pin cycles is always the coordinator worktree, never the
legacy checkout.

## Verification Commands

Use `uv run` for roadmap verification checks:

```bash
uv run python scripts/run_v3_orchestration_acceptance.py --strict
uv run python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
uv run python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json

uv run python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
uv run python src/gnn/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke
uv run python scripts/check_maintained_doc_terms.py --strict
uv run python scripts/check_repo_terminology.py --strict
uv run python scripts/check_doc_path_references.py
uv run python scripts/check_capability_contracts.py
uv run python scripts/run_semantic_fidelity_gate.py --output-dir /tmp/semantic_fidelity --strict
uv run python scripts/run_cross_framework_reliability.py --output-dir /tmp/cross_framework --strict
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file: completed items are removed when they
  land; the audit trail lives in `CHANGELOG.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.
