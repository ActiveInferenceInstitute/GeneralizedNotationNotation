# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-25 (truth-pass per the wave-6 folds at
`394400b7d`: landed rows struck — M-03 doctrine + except-Exception
ratchet landed (#203: `docs/standards/exceptions.md` + AST-pinned
`check_gnn_doc_patterns.py` ratchet at baseline 1183); M-12 MCP pymdp
gated-envelope bypass landed (#200: `run_subprocess_envelope` route +
deterministic zero-skip sandbox-refusal test); M-01 band targets
`cli/__init__.py` 1431→181 (#202), `manuscript/variables.py` 1430→562
(#201), `extract/pomdp_extractor.py` 1837→655 (#204). GEN-1/2 landed
earlier (#183, #179-#182). M-11 lean-cancel: GNN-side CancelToken
threading staged (lane t-0044); the fep-side bridge substance is held
pending the sibling fep lane (custody cycle #27 closes it). Remaining
open rows: M-01 (band remainder: `executor.py` 1665 next,
`render/pomdp_processor.py` and `render/processor.py` queued),
M-09 (mirror-or-exempt), M-14 (bnlearn seam), M-11 (fep-side held),
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
| M-01 | medium | Oversized band remainder: wave-6 folded `cli/__init__.py` 1431→181 (#202), `manuscript/variables.py` 1430→562 (#201), `extract/pomdp_extractor.py` 1837→655 (#204); next `execute/executor.py` 1665 (one file per wave), then `render/pomdp_processor.py` and `render/processor.py`; raw-extras set unchanged, pending a band-program decision; parity-gated | SCOPE §Medium M-01 |
| GEN remainder | major | Multi-agent continuous topologies (GEN-1 #183, GEN-2 #179-#182, GEN-3 factored/hybrid #193, GEN-4 non-stationary #192 all landed); partial stigmergic multi-agent support exists (`tests/render/test_stigmergic_multi_agent.py`) | SCOPE §Major |
| M-09 | medium | Mirror-or-exempt for the 7 unmirrored test dirs; `tests/parsers` + `tests/processing` mirrors dispatched in a parallel lane; near-empty src dirs (`documentation`, `grammars`, `schemas`, `schema`, `formal_specs`, `type_systems`) exemptable | SCOPE §Medium M-09 |
| M-14 | medium | bnlearn seam → `render/bnlearn/` package per the ngclearn pattern, rewire `framework_registry.py:205` + `health.py:56-58` | SCOPE §Medium |
| M-11 | medium | lean CancelToken: GNN-side threading staged (lane t-0044: executor.py lean branch + `lean_runner.verify_document` → `run_subprocess_envelope(cancel_token=...)` — real behavior change: the envelope implements process-level cancellation, the pymdp route is the pattern); fep-side in-bridge cooperative semantics HELD for the sibling fep lane; custody cycle #27 re-seals at close | SCOPE §Medium |


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
