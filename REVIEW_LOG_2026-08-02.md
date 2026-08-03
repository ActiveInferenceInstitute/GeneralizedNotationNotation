# REVIEW LOG — 2026-08-02

Mega-deep documentation review pass on `GeneralizedNotationNotation`
(ActiveInferenceInstitute public submodule). Scope: documentation accuracy,
structure, links, and code-adjacent docs. All findings below were verified
against the actual repository state; no fabricated claims.

## Preflight

- Branch: `main`; fast-forwarded from `5c72cf38` → `4db3c9fa` (18 commits).
- Repo is a git submodule (gitdir at `../../.git/modules/repos/GeneralizedNotationNotation`).
- 1,174 Markdown files tracked; `doc/` holds 610 of them.
- Baseline gates (CI on `main` was failing):
  - `check_repo_terminology.py --strict`: **10 violations** (4 in `doc/HANDOFF.md`,
    4 in `doc/uv_0.12.0_compatibility_audit.md`, 1 in
    `src/execute/activeinference_jl/setup_environment.jl`, 1 in `src/lsp/__init__.py`).
  - `mypy src/`: **2 errors** (`src/lsp/__init__.py` — pygls `attr-defined` +
    misplaced `type: ignore` on a multi-line import).
  - `docs_audit.py --strict --check-anchors`, `check_gnn_doc_patterns.py`,
    `check_maintained_doc_terms.py`: clean.
- Full test suite (command of record, Ollama files ignored): **2,622 passed,
  1 failed**. The single failure is environment-dependent:
  `test_pomdp_gridworld_cross_framework.py::test_gridworld_render_execute_analyze_visualize_strict`
  requires Julia packages (RxInfer, ActiveInference.jl) installed locally —
  documented as a prerequisite in `doc/HANDOFF.md`; not a code regression.
- Heavy suites not run: full Julia backend execution, full 25-step pipeline run,
  Bandit SARIF scan (CI-only), Ollama LLM tests (no local daemon).

## Findings & fixes by severity

### Minor (typos, formatting, metadata) — all fixed

1. `README.md` doc count "605 Markdown files under doc/" → 610 (measured).
2. `README.md` "33 top-level source/doc dirs" → "32 module directories" (measured).
3. `README.md` "output/ (ignored except .gitkeep)" → "tracked per repo policy"
   (313 files under `output/` are tracked; `.github/AGENTS.md` policy says keep them committed).
4. `README.md` duplicated v3.0.0 paragraphs consolidated ("Features (v3.0.0)"
   + "New in v3.0.0" → single "New in v3.0.0").
5. `README.md` test-evidence block updated to measured 2026-08-02 numbers
   (was: 2,495 passed + "10 uv-environment tests fail"; now: 2,622 passed / 1 env-dependent).
6. `README.md` mypy claim "4 pre-existing errors" → "clean (0 errors, 758 files)".
7. `README.md` / `AGENTS.md` / `.github/README.md` / `doc/INDEX.md` "Last updated" dates bumped.
8. `doc/HANDOFF.md` state table "Doc pages 609" → 610.
9. `doc/SETUP.md` `ActiveInference.jl` URL: `biaslab/ActiveInference.jl` (404) →
   `ComputationalPsychiatry/ActiveInference.jl` (verified 200); `rxinfer.ml` (404) →
   `docs.rxinfer.com/stable/` (verified 200).

### Medium (stale sections, wrong claims, dead links) — all fixed

1. **Terminology gate violations (CI-blocking)** — reworded banned terms in
   `doc/HANDOFF.md` (4), `doc/uv_0.12.0_compatibility_audit.md` (4), and 2 code
   comments, keeping meaning. `check_repo_terminology.py --strict` now passes.
2. **mypy gate (CI-blocking)** — moved the `# type: ignore[attr-defined]` onto the
   correct line of the multi-line `pygls.server` import and reworded the comment
   (`src/lsp/__init__.py`). `mypy src/` passes (0 errors, 758 files).
3. **Private/internal content in public docs** — removed the machine-injected
   "COGNILAYER" block (personal tool names, session-bridge/verify-identity
   workflow, session state) from `CLAUDE.md`; redacted the personal assistant
   signature in `doc/HANDOFF.md`; removed an internal staging host
   (`stg.wbdg.org`) reference from `doc/sapf/sapf.md`.
4. **Phantom optional extras** — `CLAUDE.md`, `SETUP_GUIDE.md`,
   `doc/SETUP.md`, `src/setup/AGENTS.md`, `doc/gnn/modules/01_setup.md`,
   `CHANGELOG.md` referenced `uv sync --extra llm|visualization|inference|
   active-inference|execution-frameworks` extras that do not exist in
   `pyproject.toml`. Rewrote the extras documentation to the canonical groups
   (dev, api, ml-ai, audio, gui, graphs, research, scaling, all) and clarified
   that PyMDP/JAX/NumPyro/DisCoPy are core dependencies.
5. **Framework status drift** — `doc/SETUP.md`, `doc/dependencies/
   OPTIONAL_DEPENDENCIES.md` marked PyMDP/JAX as optional-to-install; they are
   core. Updated status tables, install strategies, FAQ, and the stale
   "Pipeline v2.1.0" / "29% execution success" / "90% pass rate" claims.
   Replaced the pre-1.0 PyMDP API example with the verified pymdp 1.0.0
   JAX-first pattern (checked against the installed package and
   `src/tests/execute/test_pymdp_1_0_0_upstream_api.py`).
6. **~70 dead external links fixed** across 40+ files, each replacement
   verified live (curl): ActiveInference.jl org move (3 files), RxInfer docs
   move (5 files), MCP spec → `modelcontextprotocol.io/specification/2025-06-18`
   (2 files), DSPy docs restructure → `dspy.ai/learn/` (1 file), GEXF spec →
   `gexf.net` (2 files), dead GitHub Pages docs site → in-repo links
   (`doc/gnn/operations/gnn_tools.md`), CHANGELOG links to never-created tags
   v1.0.0/v1.1.0 → releases page, onefilellm spec template →
   `doc/gnn/reference/{0}.md`, and many single-file URL swaps
   (quadray, nockchain, iroh, timep, spm, pkl, bnlearn, performance,
   muscle-mem, glowstick, onefilellm, vec2text-adjacent, x402, deployment).
   Removed/annotated internal or retired targets (`stg.wbdg.org`, 21.co,
   langtrace blog). False positives identified and left untouched: crates.io
   and paperswithcode block automated clients; Wikipedia links with trailing
   parens were regex artifacts; localhost/example.com are configuration
   examples.
7. **`doc/HANDOFF.md`** — marked superseded, updated the commit reference,
   redacted the personal signature, fixed terminology.
8. **CHANGELOG extras line** reworded (no phantom extra reference).

### Major (structural / cross-cutting) — implemented or deferred

1. **External-link checker added** — new maintained script
   `scripts/check_external_links.py` scans all maintained docs for dead
   external URLs (concurrent, timeout, placeholder-aware, informational).
   Documented in `scripts/AGENTS.md` and `.github/README.md` local-validation.
   Deliberately NOT wired into CI (external hosts bot-block/rate-limit;
   flaky gates are worse than none).
2. **Docs index / navigation** — `doc/INDEX.md`, `doc/README.md`,
   `doc/START_HERE.md`, `doc/gnn/` hubs were already coherent and passing the
   strict audit; no restructure needed. Verified rather than churned.
3. **Deferred** (listed in `TO-DO.md`): ~20 dead citation footnotes with no
   verifiable replacement (CiteSeerX service retired, dead university hosts,
   removed PDFs). Each was checked individually; replacing them would require
   inventing citations, which is out of scope for an accuracy pass.

## Verification performed

- `docs_audit.py --strict --check-anchors --no-write`: clean (only the new
  REVIEW_LOG link, which resolves after this commit).
- `check_repo_terminology.py --strict`, `check_maintained_doc_terms.py --strict`,
  `check_gnn_doc_patterns.py --strict`: clean.
- `mypy src/ --config-file pyproject.toml`: 0 errors (758 files).
- `ruff check src/ scripts/` + `ruff format --check`: clean (scripts/ included).
- Full test suite: 2,622 passed / 1 environment-dependent failure (Julia
  packages not installed locally).
- `scripts/check_external_links.py`: exercised; all replacements re-verified
  with HTTP 200.

## Files touched

See the commit list in `TO-DO.md`; per-commit `git show --stat` is authoritative.
