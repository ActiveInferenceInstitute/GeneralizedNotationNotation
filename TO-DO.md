# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-02
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

**Last reviewed**: 2026-08-02 — mega-deep documentation review pass; see
[REVIEW_LOG_2026-08-02.md](REVIEW_LOG_2026-08-02.md) for the audit trail.

## Current 3.0.0 Status

GNN v3.0.0 is released. The long-running orchestration contracts are in place:
durable observation streams, resumable run sessions, and auditable container
plans. Current local catch-up checks pass for the strict v3.0.0 acceptance gate,
run-manifest emission from `output/`, and container-plan generation from
`input/config.yaml`. The session-wrapped all-family acceptance path has also
been exercised with 9 of 9 families marked `DONE` and no failed units.

This roadmap is forward-only. Shipped-version history belongs in `CHANGELOG.md`,
release notes, and verification artifacts, not in this open-work queue.

## 2026-08-02 Documentation Review

Completed mega-deep documentation review (relative links, external links,
accuracy vs. code, dependency-group claims, CI-gate health, private-content
hygiene). Items below are the scoped findings; ✓ marks completed work with the
commit reference. The full audit trail is in
[REVIEW_LOG_2026-08-02.md](REVIEW_LOG_2026-08-02.md).

### Minor (typo, broken link, formatting, metadata)

- ✓ `README.md`: doc/ Markdown count 605 → 610 (measured) — `8fd456f7`
- ✓ `README.md`: "33 top-level source/doc dirs" → "32 module directories" — `8fd456f7`
- ✓ `README.md`: `output/` "ignored except .gitkeep" → "tracked per repo policy" — `8fd456f7`
- ✓ `README.md`: consolidated duplicate v3.0.0 paragraphs — `8fd456f7`
- ✓ `README.md` / `AGENTS.md` / `.github/README.md` / `doc/INDEX.md`: "Last updated" dates — `8fd456f7`
- ✓ `doc/HANDOFF.md`: state-table doc-page count 609 → 610 — `1931fb52`
- ✓ `doc/SETUP.md`: `biaslab/ActiveInference.jl` (404) → `ComputationalPsychiatry/ActiveInference.jl`; `rxinfer.ml` → `docs.rxinfer.com/stable/` — `fb4898db`

### Medium (stale section rewrite, doc restructure, added missing guide)

- ✓ CI-gate repair: `check_repo_terminology.py --strict` 10 violations → 0
  (`doc/HANDOFF.md` ×4, `doc/uv_0.12.0_compatibility_audit.md` ×4, 2 code
  comments) — `1931fb52`
- ✓ CI-gate repair: `mypy src/` 2 errors → 0 (pygls `type: ignore` placement in
  `src/lsp/__init__.py`) — `1931fb52`
- ✓ Private-content hygiene: removed CogniLayer block from `CLAUDE.md`;
  redacted personal assistant signature in `doc/HANDOFF.md`; removed internal
  staging host (`stg.wbdg.org`) from `doc/sapf/sapf.md` — `f1b3b3bc`,
  `1931fb52`, `fb4898db`
- ✓ Phantom optional extras corrected across `CLAUDE.md`, `SETUP_GUIDE.md`,
  `doc/SETUP.md`, `doc/dependencies/OPTIONAL_DEPENDENCIES.md`,
  `src/setup/AGENTS.md`, `doc/gnn/modules/01_setup.md`, `CHANGELOG.md` — `f1b3b3bc`
- ✓ Framework status tables updated (PyMDP/JAX/NumPyro/DisCoPy = core; Julia
  backends optional; PyTorch manual) — `f1b3b3bc`
- ✓ `~70` dead external links fixed across 40+ files, each replacement
  verified live (see REVIEW_LOG for the full list) — `fb4898db`
- ✓ `doc/HANDOFF.md`: marked superseded, commit reference and terminology
  corrected — `1931fb52`
- ✓ `README.md` / `AGENTS.md`: validation evidence refreshed with measured
  2026-08-02 numbers (2,622 passed / 1 environment-dependent Julia failure;
  mypy clean) — `8fd456f7`

### Major (large doc system overhaul, new documentation site, cross-cutting refactors)

- ✓ Added maintained external-link checker
  `scripts/check_external_links.py` (informational; deliberately not CI-wired
  because external hosts bot-block/rate-limit) — `4f2ecdf2`
- Open — Deferred: ~20 dead citation footnotes with no verifiable replacement
  (CiteSeerX retired, removed PDFs on dead university hosts, deleted blog
  posts). Replacing them would require inventing citations; left as-is and
  enumerated in REVIEW_LOG §Findings (Major item 3). Fix by locating
  authoritative replacements (DOI/arXiv) per citation.
- Open — Deferred: full CI parity on this machine requires Julia backend
  packages (RxInfer, ActiveInference.jl) and an Ollama daemon; the one failing
  test is `test_pomdp_gridworld_cross_framework.py::test_gridworld_render_execute_analyze_visualize_strict`
  (environment-dependent, documented in `doc/HANDOFF.md`).

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
- Keep closed work out of this file (the 2026-08-02 review section above is the
  documented exception: completed items carry ✓ + commit refs and are retained
  only as the audit-visible close-out of that pass).
