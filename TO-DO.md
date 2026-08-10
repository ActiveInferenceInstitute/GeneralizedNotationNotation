# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-07
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

**Last reviewed**: 2026-08-07 — RxInfer native-strategy campaign + corpus
curation + doc/gnn accuracy review. The full audit trail lives in
`CHANGELOG.md` ([Unreleased]) and git history; RxInfer-specific state and
open items live in [RXINFER_IMPROVEMENT_ROADMAP.md](RXINFER_IMPROVEMENT_ROADMAP.md).

## Current 3.0.0 Status

All gates and the full pipeline were exercised locally on 2026-08-07, on the
curated 29-exemplar corpus (46 → 29; regenerable scaling-sweep artifacts and
one redundant fixture pruned, −67 MB):

- Full test suite: **2,797 passed / 0 failed / 2 skipped** (both allowlisted
  environment-gated files; 2,799 collected; Ollama files ignored per the
  command of record).
- Full pipeline (`src/main.py --target-dir input/gnn_files`): **25/25 steps,
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
- Doc gates: docs_audit (anchors included) 0 issues; doc patterns and
  maintained terms clean; repo terminology count reduced (48 → 43 during the
  doc review; residual violations are deliberate deprecation wording).

This roadmap is forward-only. Shipped-version history belongs in
`CHANGELOG.md`, release notes, and verification artifacts.

## Open Work

- **RxInfer-specific open items** — tracked in
  [RXINFER_IMPROVEMENT_ROADMAP.md](RXINFER_IMPROVEMENT_ROADMAP.md): N-level
  native hierarchical rendering (decision recorded: joint composition until
  exemplars declare composed coupling), dashboard real-browser verification
  pass, optional T=100 precompile workloads if batches become routine.
- **Orchestrator line-count drift** — the per-step docs cite exact
  orchestrator line counts with no gate checking them; 14 of 25 had rotted
  silently before the 2026-08-07 doc review corrected them. Decide: add a
  gate under `scripts/` that verifies the documented counts, or drop exact
  numbers in favor of the thin-orchestrator invariant (<150 lines).
  Acceptance: either the gate exists and passes in the default suite, or all
  module docs carry the invariant instead of literals.
- **Type-checker `.gnn` discovery** — `src/type_checker/checking/core.py`
  discovers only `*.md`, while the parser stack supports `.gnn` via
  `get_supported_gnn_extensions()`. Either extend discovery to the supported
  extensions or document `.md` as the only supported spec extension
  everywhere (the quickstart tutorial was corrected to `.md` meanwhile).
  Acceptance: a `.gnn` fixture is either type-checked or explicitly rejected
  with a clear message, plus a pinning test.
- **Doc version banners** — 40 files under `doc/gnn/**` carry
  "**Version**: v1.6.0 Engine"; actual engine version is 3.0.0
  (`pyproject.toml`). Sweep them in one pass (kept out of the per-partition
  doc review to avoid cross-partition churn). Acceptance: no `v1.6.0 Engine`
  banner remains; a grep-based check added to the docs audit if cheap.

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
