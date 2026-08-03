# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-02
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

**Last reviewed**: 2026-08-02 — mega-deep documentation review pass; both
scoped follow-up workstreams (dead-link remediation and full CI parity) are
**completed** and removed from this file per the conventions below. The full
audit trail — findings, fixes, verification — is in
[REVIEW_LOG_2026-08-02.md](REVIEW_LOG_2026-08-02.md).

## Current 3.0.0 Status

GNN v3.0.0 is released. The long-running orchestration contracts are in place:
durable observation streams, resumable run sessions, and auditable container
plans. All CI gates and the full pipeline have been exercised locally on
2026-08-02:

- Full test suite: **2,658 passed / 0 failed / 0 skipped** (no ignores;
  `JULIA_PROJECT=/tmp/julia_test_env`; Ollama `smollm2:135m-instruct-q4_K_S`).
- `run_v3_orchestration_acceptance.py --strict`: **19/19 checks passed**.
- Full pipeline (`src/main.py --target-dir input/gnn_files`, isolated
  output): **25/25 steps, 0 failed, 100% success rate**.
- `uv lock --check`, capability contracts, manuscript tokens, pomdp-outputs
  check, semantic fidelity gate, bandit (0 Medium/High), run-manifest
  emission, container plan (0 findings): all clean.
- v4.0.0 autonomous smoke: 3 proposal-only candidates, no mutations.

This roadmap is forward-only. Shipped-version history belongs in `CHANGELOG.md`,
release notes, and verification artifacts, not in this open-work queue.

## Open Work

Nothing is open in this roadmap at this time. Both scoped follow-up
workstreams from the 2026-08-02 review are complete (see
`REVIEW_LOG_2026-08-02.md` §Follow-up implementation and §Full CI-gate
parity). Future items (e.g. the next docs sweep, dependency upgrades, or
v4.0.0 implementation work) should be scoped here with concrete tasks, file
paths, verification commands, and acceptance criteria.

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface now emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container run,
or cluster mutation is automatic.

No additional v4.0.0 implementation item is open in this roadmap at this time.

## Verification Commands

Use `uv run --frozen` for roadmap catch-up checks until `uv.lock` is deliberately
refreshed.

```bash
PYTHONPATH=src uv run --frozen python scripts/run_v3_orchestration_acceptance.py --strict
PYTHONPATH=src uv run --frozen python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
PYTHONPATH=src uv run --frozen python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json
PYTHONPATH=src uv run --frozen python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
PYTHONPATH=src uv run --frozen python src/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run --frozen --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --extra dev python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --extra dev python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --extra dev python scripts/check_repo_terminology.py --strict
uv run --frozen --extra dev python scripts/check_external_links.py   # informational
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file: completed items are removed when they
  land; the audit trail lives in `REVIEW_LOG_2026-08-02.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.
