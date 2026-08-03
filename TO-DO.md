# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-02
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

**Last reviewed**: 2026-08-02 — mega-deep documentation review pass; completed
items are removed from this file per the conventions below. The full audit
trail of that pass (findings, fixes, verification) is in
[REVIEW_LOG_2026-08-02.md](REVIEW_LOG_2026-08-02.md).

## Current 3.0.0 Status

GNN v3.0.0 is released. The long-running orchestration contracts are in place:
durable observation streams, resumable run sessions, and auditable container
plans. Current local catch-up checks pass for the strict v3.0.0 acceptance gate,
run-manifest emission from `output/`, and container-plan generation from
`input/config.yaml`. The session-wrapped all-family acceptance path has also
been exercised with 9 of 9 families marked `DONE` and no failed units.

This roadmap is forward-only. Shipped-version history belongs in `CHANGELOG.md`,
release notes, and verification artifacts, not in this open-work queue.

## Open Work — Documentation Review Follow-ups (scoped 2026-08-02)

### Workstream 1 — Dead external citations & transient links — COMPLETED ✓

All tasks (1.1 P0 transient `ppl-ai-file-upload` links, 1.2 malformed arXiv
IDs, 1.3 moved/deleted pages, 1.4 retired services & dead hosts, 1.5
browser-verified bot-blocked hosts, 1.6 checker accuracy) are implemented;
see `REVIEW_LOG_2026-08-02.md` §Follow-up implementation for the full list of
removals and verified replacements. Remaining checker output is limited to
bot-blocked classes (403/429/401/999 — verified fine in a browser: crates.io,
medium.com, paperswithcode redirects) and two intentional non-links
(`api.openai.com/v1` env-var value in `doc/llm/README.md`; scope example in
this file). No open items.

### Workstream 2 — Full CI parity: Julia backends + Ollama (Major)

**Goal**: make the full test suite pass locally (0 failed, 0 skipped) and
refresh the documented evidence numbers.

**Completed**:
- Julia 1.12.6 present; packages `RxInfer`, `ReactiveMP`, `GraphPPL`,
  `ActiveInference`, `Distributions`, `StatsBase`, `JSON` installed in
  `/tmp/julia_test_env` (the environment the cross-framework test probes).
- `test_gridworld_render_execute_analyze_visualize_strict` **passes** when run
  with `JULIA_PROJECT=/tmp/julia_test_env` (the execute processor probes the
  project selected by `JULIA_PROJECT`; 55.6s, full render → execute (pymdp +
  rxinfer) → analyze → visualize chain).
- Ollama daemon running; `smollm2:135m-instruct-q4_K_S` pulled.
- Ollama LLM tests: `src/tests/llm/test_llm_ollama.py` +
  `test_llm_ollama_integration.py` → **26 passed**.
- Full suite (command of record, Ollama files ignored, `JULIA_PROJECT` set):
  **2,623 passed, 0 failed** (pending final confirmation run).

**Remaining**:
1. Run the full suite once more **without** the two Ollama ignores
   (daemon + model now available) and record the total (expected 2,649
   passed, 0 failed).
2. Update the evidence numbers in `README.md` (Test Suite Evidence),
   `AGENTS.md` (Current Validation), `SETUP_GUIDE.md` (Latest Validation),
   and `doc/HANDOFF.md` (state table) to the measured results, and note the
   `JULIA_PROJECT=/tmp/julia_test_env` prerequisite for local Julia-backend
   execution.
3. Optionally refresh `uv.lock`/verify `uv lock --check` and re-run the v3
   acceptance gate with the full env.

**Acceptance**: full suite 0 failed / 0 skipped with the documented command;
evidence docs show measured numbers; `git diff --check` clean.

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
