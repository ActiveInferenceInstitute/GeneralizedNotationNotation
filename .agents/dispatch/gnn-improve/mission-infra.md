# Infrastructure & Utils Scope (setup, template, tests infra, doc, pipeline) — mission-infra.md

You own these paths ONLY within the GNN repo at
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`:

- src/setup/
- src/template/
- src/tests/infrastructure/   (env/utils/infra test mirror)
- src/utils/                   (shared helpers; you are the ONLY owner)
- src/pipeline/                (orchestration contracts: durable_streams,
  run_session, container_plan, session_acceptance, run_manifest,
  pipeline_container_plan)
- src/lsp/
- src/sapf/ (public entry; impl lives under src/audio/sapf)
- doc/                        (technical doc subtree; strict docs build)
- doc's gates: doc/development/docs_audit.py, scripts/check_gnn_doc_patterns.py,
  scripts/check_maintained_doc_terms.py, scripts/check_repo_terminology.py

DO NOT touch: pyproject.toml, justfile, uv.lock, pytest.ini, README.md,
AGENTS.md, CLAUDE.md, CHANGELOG.md, DOCS.md, manuscript/, input/, output/.

GOAL (shallow→deep):
1. Utils robustness — fix real edge cases in arg parsing, logging, file
   helpers. Keep the shared utils API surface stable. Add missing edge
   tests.
2. Pipeline orchestration (v3.0.0 contracts): make the durable
   streams / resumable run-session / auditable container-plan contracts
   fail closed and be replayable. Verify existing acceptance gates:
   `uv run --extra dev python scripts/run_v3_orchestration_acceptance.py`
3. Docs accuracy: reconcile DOCS_TO_IMPROVE.md, fix stale paths/anchors so
   `--strict` doc audit + doc-pattern + terminology gates all pass.
4. Setup/template hygiene: dependency diagnostics must report cleanly.

VERIFY (scoped):
- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write`
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict`
- `uv run --extra dev python scripts/check_maintained_doc_terms.py --strict`
- `uv run --extra dev python scripts/check_repo_terminology.py --strict`
- `uv run --extra dev python scripts/run_v3_orchestration_acceptance.py`
- `uv run --extra dev python -m pytest src/tests/infrastructure src/tests/setup src/tests/template src/tests/pipeline -q --tb=no -x`
- `uv run ruff check src/utils src/pipeline src/setup src/template` (+ format)

If the full docs audit was already green, add targeted improvements only
where genuinely missing; do not pad.

HARD RULE: leave ALL changes uncommitted; no commit/push.

## Finish
Write a concise report to
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-improve/REPORT-infra.md`
Summarize files changed, docs fixed, acceptance-gate results. Reply with
only that report's absolute path.