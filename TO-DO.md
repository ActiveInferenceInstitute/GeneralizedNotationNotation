# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-17 (clean-start execution correctness; `-n auto` fully green)
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

**Last reviewed**: 2026-08-18 — all RED_TEAM_REVIEW.md security residuals V-01–V-11 closed, `-n auto` fully green with **3,039 passed / 0 failed / 0 skipped** on a fully provisioned environment (D2 CLI, Julia RxInfer/StatsBase backends, a local Ollama daemon with `smollm2:135m-instruct-q4_K_S`, and `RANDOM_SIMULATION_ENABLED=1`), the three HANDOFF §4 test-infrastructure follow-ups resolved, and three clean-start execution regressions fixed: Julia frameworks now execute from their committed `--project` environments (no longer silently skipped against the global depot); the render manifest aggregates across per-folder pipeline invocations; and Step 12 execution is scoped to the current folder, so every rendered model is executed exactly once instead of only the last folder's model (or, post-aggregation, every folder's models ten times). `test_uv_sync_fast` also retries transient `uv sync --check` "outdated" races.
`doc/ → doc/archive/` reorganization, `argument_utils` + RxInfer strategy
modularization, and public-API test coverage. The full audit trail lives in
`CHANGELOG.md` ([Unreleased]) and git history; RxInfer-specific state and
open items live in [RXINFER_IMPROVEMENT_ROADMAP.md](RXINFER_IMPROVEMENT_ROADMAP.md).

## Current 3.0.0 Status

All gates and the full pipeline were exercised locally on 2026-08-07, on the
curated 29-exemplar corpus (46 → 29; regenerable scaling-sweep artifacts and
one redundant fixture pruned, −67 MB):

- Full test suite: **2,797 passed / 0 failed / 2 skipped** (both allowlisted
  environment-gated files; 2,799 collected; Ollama files ignored per the
  command of record).
- Full pipeline (`src/main.py --target-dir input/gnn_files --output-dir /tmp/gnn-full-pipeline`): **25/25 steps,
  0 failed** (2 warnings: optional-viz assets and the deliberately-absent
  Playwright PNG path), 1h28m. Step 12 executes every model under every
  installed backend — including ActiveInference.jl from its committed
  minimal environment for the first time (the old committed env could never
  build on a clean machine).
- Every ModelKind renders natively (flat batch/online, hierarchical
  two-level, factored, continuous LGSSM, Dirichlet learning) or via the
  documented joint composition (multi-agent, 3+-level hierarchical); every
  kind live-executed `all_valid=true`.
- M8 GIF batch: 29/29 clean at T=100 with reproducibility manifests;
  dashboard regenerated (category + state-size filters, compare mode).
- `run_v3_orchestration_acceptance.py --strict`: 19/19. Container plan: 0
  findings. Model-family acceptance: green against the updated manifest.
- Doc gates: docs_audit (anchors included), doc patterns, maintained terms,
  and repository terminology are clean.

This roadmap is forward-only. Shipped-version history belongs in
`CHANGELOG.md`, release notes, and verification artifacts.

## Open Work

- **RxInfer-specific open items** — tracked in
  [RXINFER_IMPROVEMENT_ROADMAP.md](RXINFER_IMPROVEMENT_ROADMAP.md): N-level
  native hierarchical rendering (decision recorded: joint composition until
  exemplars declare composed coupling), and optional T=100 precompile
  workloads if batches become routine. The dashboard real-browser verification
  pass (screenshot + DOM + a11y) is now complete (2026-08-18).

### Security follow-ups (residual from the 2026-08-14 red-team wave)

The 2026-08-14 remediation wave and its wave-2 follow-up (2026-08-14) closed
all RED_TEAM_REVIEW.md items: V-03 `safe_literal_eval` migration is complete
across all nine remaining files, V-01 Julia pre-execution gating is blocking
(`Base.Meta.parseall`), V-10 manifest-based script discovery is implemented,
and V-11 FastAPI rate limiting is live (`api.rate_limit`, `GNN_RATE_LIMIT`).
No unchecked security residuals remain at this time.

### Test-infrastructure follow-ups (from doc/HANDOFF.md §4 — resolved 2026-08-17)

- **Parallel test execution.** `-n auto` is now fully green: **3,039 passed /
  0 failed / 0 skipped** with D2 CLI, Julia RxInfer/StatsBase backends, Ollama,
  and `RANDOM_SIMULATION_ENABLED=1` provisioned (the two Ollama files now run
  instead of being ignored). The two residual failures are closed: the strict GridWorld cross-framework test now
  skips cleanly when Julia backend packages are absent (cached availability
  probe), and `get_installed_package_versions` retries transient subprocess /
  incomplete-enumeration races while `test_uv_sync_fast` uses non-mutating
  `uv sync --check` so the shared `.venv` is never rewritten concurrently.
- **Type-annotation completion.** No real source function lacks annotations:
  `mypy` passes with `disallow_untyped_defs` + `disallow_incomplete_defs`
  across 812 files. The remaining untyped-looking signatures live inside
  string-embedded generated-code templates (`src/render/*/templates/*`,
  DisCoPy/JAX renderers), which are output text, not callable source.
- **Standalone test files.** Decision recorded: pin as documentation-embedded
  examples, not pytest tests. Six files remain under `doc/` (activeinference_jl,
  cognitive_phenomena, pymdp); they are `unittest`/standalone scripts with
  doc-local imports and are already outside `testpaths` (`src/tests`, `tests`).
  `src/llm/test_llm_system.py` was removed in commit `40068ba4`.

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container
run, or cluster mutation is automatic. No additional v4.0.0 implementation
item is open in this roadmap at this time.

## Verification Commands

Use `uv run --frozen` for roadmap catch-up checks until `uv.lock` is
deliberately refreshed.

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
  land; the audit trail lives in `CHANGELOG.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.
