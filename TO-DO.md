# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-24 (truth-pass per the wave-5 census at
`ac8abd3a2`: landed rows struck — Minor batch MI-1..MI-16 (#173-#176),
BC-02a/BC-02c + ARCH-1, BC-02b, BC-13, ARCH-3, and
M-04/M-05/M-06/M-07/M-08/M-10/M-13; GEN-1/2 landed (#183, #179-#182).
M-01 re-counted to the tracked band set: 11 after #188 folded
`main.py` (a raw >1200-line sweep counts 17). Open rows re-scoped:
M-03 (doctrine long tail), M-09 (mirror-or-exempt), M-11/M-12/M-14,
BC-14 (conditional). Audit trail: `CHANGELOG.md` and git history.
Full program: `SCOPE-2026-09-23.md`.)
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
| BC-14 | minor | `utils/pipeline_orchestration/pipeline_template.py` I5 fallback chain — conditional on lint-imports going red (currently 3 kept / 0 broken) | SCOPE §Medium M-06 |
| M-01 | medium | Oversized band: 11 tracked band files (SCOPE M-01 set + new crossers) after #188 folded `main.py` (largest `rxinfer_bridge.py` 1801 … `render/processor.py` 1329; new crossers `execute/processor.py` 1577, `website/generator.py` 1379); a raw >1200-line sweep counts 17 at `9fb81279e` — 6 extras sit outside the tracked set (`gui/gui_2/ui.py` 1545, `parsers/schema_parser.py` 1534, `utils/arguments/arg_parsing.py` 1498, `intelligent_analysis/processor.py` 1409, `utils/logging/logging_utils.py` 1335, `security/processor.py` 1235) pending a band-program decision; decompose one file per wave, parity-gated | SCOPE §Medium M-01 |
| GEN-3/GEN-4 | major | v4.0.0 generalization remainder: factored/hybrid/multi-agent continuous; non-stationary F_t/regime semantics (GEN-1/2 landed: #183, #179-#182) | SCOPE §Major |
| M-03 | medium | `except Exception` long tail: 304+ remaining sites, all spot-checked sites log + structured receipts; hotspot fixes landed — remaining work is doctrine + optional ratchet, no mass rewrite | SCOPE §Medium M-03 |
| M-09 | medium | Mirror-or-exempt for the 7 unmirrored test dirs; `tests/parsers` + `tests/processing` mirrors dispatched in a parallel lane; near-empty src dirs (`documentation`, `grammars`, `schemas`, `schema`, `formal_specs`, `type_systems`) exemptable | SCOPE §Medium M-09 |
| M-11/M-12/M-14 | medium | Execution-route normalization: lean CancelToken holdout (cross-repo fep lane); MCP pymdp gated-envelope bypass (rides the GEN-3/4 fold); bnlearn seam → `render/bnlearn/` package per the ngclearn pattern, rewire `framework_registry.py:205` + `health.py:56-58` | SCOPE §Medium |


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
uv run python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
uv run python src/gnn/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run python scripts/check_gnn_doc_patterns.py --strict
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
