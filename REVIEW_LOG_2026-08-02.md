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

## Follow-up implementation (2026-08-02, same pass)

The deferred items and the two scoped workstreams were implemented
comprehensively in a follow-up:

### Workstream 1 — dead citations & transient links (COMPLETED)

- **P0 hygiene**: removed all `ppl-ai-file-upload.s3.amazonaws.com` paste
  links (doc/archive/nock/jock/jock.md ×2, doc/archive/nock/nockchain/nockchain.md ×2);
  repointed the Jock-source footnotes at the live `zorp-corp/jock-lang` repo
  (merged duplicate footnote in jock.md).
- **P1 malformed IDs**: removed orphaned footnotes `21.3.00848`
  (arc-agi) and `2406.1.3.0` (vec2text ×2); swapped vec2text [18]
  `arxiv.org/html/2411.05034` → `arxiv.org/abs/2411.05034` (Eguard, verified
  200, title matches inline claim).
- **P1 moved pages**: removed orphaned dead DSPy doc footnotes ([57], [114];
  [60] verified live and kept), timep `TESTS/OUTPUT/out.profile`; swapped
  `turing.ml` docs → `turinglang.github.io/Turing.jl/stable/` (×2, verified);
  fixed d2 `releases.[1]` glued-footnote artifact; cline quickstart →
  `docs.cline.bot/` (previous commit).
- **P2 retired services & dead hosts**: CiteSeerX ×3 — one referenced
  (quadray [18], Nystrom IVM paper) replaced with verified DOI
  `10.1007/11428862_181` (Springer ICCS 2005, resolves 200), two orphaned
  removed; core.ac.uk PDF (orphaned) removed; nms.kcl.ac.uk Simeone PDF →
  Wayback (2022 snapshot); archive.org items (orphaned) removed; dead
  university hosts — brainimaging.waisman.wisc.edu ×2, brainresearch.de
  (→ UCL SPM docs, verified), lacasa.uah.edu (→ Wayback), pcl.sitehost.iu.edu,
  pls-lab.org, papl.cs.brown.edu, web.eecs.umich.edu (all orphaned, removed);
  Kirby Urner academia.edu pages ×3 → `grunch.net/synergetics/quadrays.html`
  (verified 200); OpenVINO docs → `github.com/openvinotoolkit/open_model_zoo`;
  dead domains removed (mathaware.org, zora.uzh.ch, relidator.com, icodrops,
  forum.nockchain.org, hub.athina.ai → Wayback, docfork llms.txt →
  onefilellm repo, juejin → onefilellm requirements.txt, kdjingpai, aidoczh,
  arcprize.kongjiang.org → `arcprize.org/` (×2), zorp-corp/nockapp,
  hyper.ai 500 (orphaned), PFW writing-guide link dropped); cosmometry.net →
  Wayback; darreljarmusch resume → Wayback (browser-confirmed 404 on the
  live URL).
- **P3 browser verification**: crates.io `/crates/iroh` loads fine in a real
  browser (bot-blocking confirmed — all 8 crates.io links left intact);
  paperswithcode paper pages redirect to huggingface.co/papers and 404 there —
  the 6 PWC footnotes were resolved by merging/removing (5 orphaned, 1 merged
  into the identical Eguard arXiv citation); medium/direct.mit/dl.acm/
  stackoverflow/sourceforge/lib.rs 403s are paywall/bot classes, left intact.
- **P4/P5 checker**: bot-blocked statuses (401/403/429/999) now bucketed into
  one summary line, listed only with `--strict`; regression tests added
  (`src/tests/test_check_external_links.py`, 9 tests) covering backtick
  stripping, paren re-balancing, and template skipping (caught and fixed a
  real `<your-username>` skip gap).

Remaining checker output after the sweep: 0 dead-link findings besides two
intentional non-links (`api.openai.com/v1` env-var value in doc/llm/README.md;
a scope example that has since been removed from TO-DO.md) and the
bot-blocked bucket.

### Workstream 2 — CI parity: Julia backends + Ollama (COMPLETED)

- Julia packages installed in `/tmp/julia_test_env` (RxInfer, ReactiveMP,
  GraphPPL, ActiveInference, Distributions, StatsBase, JSON); the strict
  cross-framework GridWorld test now **passes** (55.6s) when run with
  `JULIA_PROJECT=/tmp/julia_test_env` (the execute processor probes the
  `JULIA_PROJECT`-selected environment).
- Ollama daemon running; `smollm2:135m-instruct-q4_K_S` pulled; both Ollama
  test files pass (**26 passed**).
- Full suite (command of record with the two Ollama ignores, `JULIA_PROJECT`
  set): **2,632 passed, 0 failed**.
- Evidence docs (`README.md`, `AGENTS.md`, `SETUP_GUIDE.md`) updated to the
  measured numbers; `doc/HANDOFF.md` banner notes the follow-up completion.

## Verification performed (follow-up)

- `docs_audit.py --strict --check-anchors --no-write`: clean (only the new
  REVIEW_LOG link, which resolves after this commit).
- `check_repo_terminology.py --strict`, `check_maintained_doc_terms.py --strict`,
  `check_gnn_doc_patterns.py --strict`: clean.
- `mypy src/ --config-file pyproject.toml`: 0 errors (758 files).
- `ruff check src/ scripts/` + `ruff format --check`: clean (scripts/ included).
- `scripts/check_external_links.py`: exercised; all replacements re-verified
  with HTTP 200.
- Full test suite (no ignores, `JULIA_PROJECT=/tmp/julia_test_env`, Ollama
  `smollm2:135m-instruct-q4_K_S`): **2,658 passed / 0 failed / 0 skipped**;
  command-of-record suite (Ollama files ignored): **2,632 passed / 0 failed /
  0 skipped**. Ollama LLM files: **26 passed**.

### Full CI-gate parity (2026-08-02, all exercised locally)

- `uv lock --check`: resolved 310 packages, clean.
- `scripts/run_v3_orchestration_acceptance.py --strict`: **19/19 checks
  passed** (durable streams, run sessions, container plans).
- `scripts/check_capability_contracts.py`: capability contracts verified.
- `scripts/check_manuscript_tokens.py --strict`: clean (38 tokens, 14 bib
  keys).
- `scripts/check_pomdp_gridworld_outputs.py output`: clean across render,
  execute, analysis, report, and website outputs.
- `bandit -r src -c pyproject.toml --severity-level medium`: **0 Medium /
  0 High** (9 Low, pre-existing benign: subprocess imports with `nosec`,
  B105 false positives on example strings such as `hunter2`/`human-reviewed`
  in test/example code).
- `gnn preflight`: 19/20 — the single warning is `torch` absent, which is
  intentional (GHSA-rrmf-rvhw-rf47 has no patched torch release; the lock
  omits torch and bnlearn's pgmpy per `pyproject.toml`). CLAUDE.md,
  SETUP_GUIDE.md, and doc/SETUP.md all document torch as manual-optional —
  docs verified accurate.
- `gnn health`: 15/16 (same intentional torch warning); 9/9 generator modules
  importable.
- `scripts/emit_run_manifest.py output`: trace_integrity_ok True,
  re-validation clean (0 problems).
- `scripts/generate_pipeline_container_plan.py`: security review clean
  (0 findings).
- `scripts/run_semantic_fidelity_gate.py --strict`: passed.

### Full pipeline run (definitive docs-claims validation)

`python src/main.py --target-dir input/gnn_files --output-dir /tmp/gnn_full_run`
with `JULIA_PROJECT=/tmp/julia_test_env`: **25/25 steps, 0 failed, 100%
success rate, 12m total**. Three SUCCESS_WITH_WARNINGS, all expected:
step 9 Playwright driver install (browser-viz opt-in surface), step 12
ActiveInference.jl absent from the Julia env at run time (pymdp + rxinfer
executed; the package was then installed and patched per
`src/execute/activeinference_jl/setup_environment.jl` — the full
`using JSON, Distributions, StatsBase, RxInfer, ActiveInference` probe now
passes, so a re-run has no Julia warning), step 24 smollm2 hallucinated code →
robust rule-based fallback (graceful by design). The v4.0.0 autonomous smoke
(`--autonomous`) wrote 3 proposal-only candidates under
`/tmp/gnn-autonomous-smoke/autonomous` with no source edits or mutations.

## Files touched

See the commit list in `TO-DO.md`; per-commit `git show --stat` is authoritative.
