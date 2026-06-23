# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-06-23
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

## Current 3.0.0 Status

GNN v3.0.0 is released. The long-running orchestration contracts are in place:
durable observation streams, resumable run sessions, and auditable container
plans. Current local catch-up checks pass for the strict v3.0.0 acceptance gate,
run-manifest emission from `output/`, and container-plan generation from
`input/config.yaml`. The session-wrapped all-family acceptance path has also
been exercised with 9 of 9 families marked `DONE` and no failed units.

This roadmap is forward-only. Shipped-version history belongs in `CHANGELOG.md`,
release notes, and verification artifacts, not in this open-work queue.

## Open Work

No open roadmap items are currently tracked. Add new items here only when there
is forward-looking work that is not already represented by release notes,
verification artifacts, or current implementation status.

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
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file.
